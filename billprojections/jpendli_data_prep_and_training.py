from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import DecimalType
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime
from decimal import Decimal
import holidays
import math
import re
import logging
import os
import pandas as pd
import pickle

# import mtr_list

# import mtr_list

s3_warehouse_path = "s3://entergy-bdi-dataeng-code-repo-training/entergy-bdi-etl/jpendli/datascience/bill_projection/"  # s3 bucket for raw layer
# s3_warehouse_path = "s3://entergy-bdi-dq-dev/"

# label path to folder where the model will be stored
folder_name = "models/model_2"
model_store_path = "[s3 path to bill projection folder]"+ folder_name

spark = (SparkSession
         .builder
         .appName("test")
         .config('spark.sql.extensions', 'org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions')
         .config('spark.sql.catalog.glue_catalog', 'org.apache.iceberg.spark.SparkCatalog')
         .config('spark.sql.catalog.glue_catalog.catalog-impl', 'org.apache.iceberg.aws.glue.GlueCatalog')
         .config('spark.sql.catalog.glue_catalog.io-impl', 'org.apache.iceberg.aws.s3.S3FileIO')
         .config('spark.sql.catalog.glue_catalog.warehouse', s3_warehouse_path)
         .config("spark.sql.catalog.glue_catalog.lock.table", "myIcebergLockTable")
         .getOrCreate()
         )
sc = spark.sparkContext
sc.setLogLevel("ERROR")

location = './data/'
filename = 'training_data.pkl'


def write_data(df, location, filename):
    df.to_pickle(location + filename)


# define distance UDF after spark session started
@F.udf(returnType=DecimalType(16, 10))
def calculate_distance(lat1, lon1, lat2, lon2):
    # Vincenty's distance formula, emulating SAS GEODIST
    # Convert coordinates to radians
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = math.radians(lat1), math.radians(lon1), math.radians(lat2), math.radians(
        lon2)

    # Vincenty's formula
    a = 6378137  # Semi-major axis of the Earth in meters
    b = 6356752.314245  # Semi-minor axis of the Earth in meters
    f = 1 / 298.257223563  # Flattening of the Earth
    L = lon2_rad - lon1_rad

    U1 = math.atan2((1 - f) * math.sin(lat1_rad), math.cos(lat1_rad))
    U2 = math.atan2((1 - f) * math.sin(lat2_rad), math.cos(lat2_rad))
    sinU1, cosU1 = math.sin(U1), math.cos(U1)
    sinU2, cosU2 = math.sin(U2), math.cos(U2)

    lambda_old = L
    iter_limit = 100  # Maximum number of iterations
    iter_count = 0  # Iteration counter

    while True:
        sin_lambda, cos_lambda = math.sin(lambda_old), math.cos(lambda_old)
        sin_sigma = math.sqrt((cosU2 * sin_lambda) ** 2 + (cosU1 * sinU2 - sinU1 * cosU2 * cos_lambda) ** 2)
        cos_sigma = sinU1 * sinU2 + cosU1 * cosU2 * cos_lambda
        sigma = math.atan2(sin_sigma, cos_sigma)
        sin_alpha = cosU1 * cosU2 * sin_lambda / sin_sigma
        cos_sq_alpha = 1 - sin_alpha ** 2
        cos2_sigma_m = cos_sigma - 2 * sinU1 * sinU2 / cos_sq_alpha

        C = f / 16 * cos_sq_alpha * (4 + f * (4 - 3 * cos_sq_alpha))
        lambda_new = L + (1 - C) * f * sin_alpha * (
                sigma + C * sin_sigma * (cos2_sigma_m + C * cos_sigma * (-1 + 2 * cos2_sigma_m ** 2)))

        # Check convergence (difference between lambda_new and lambda_old)
        if abs(lambda_new - lambda_old) < 1e-6 or iter_count >= iter_limit:
            break

        lambda_old = lambda_new
        iter_count += 1

    u_sq = cos_sq_alpha * (a ** 2 - b ** 2) / (b ** 2)
    A = 1 + u_sq / 16384 * (4096 + u_sq * (-768 + u_sq * (320 - 175 * u_sq)))
    B = u_sq / 1024 * (256 + u_sq * (-128 + u_sq * (74 - 47 * u_sq)))
    delta_sigma = B * sin_sigma * (
            cos2_sigma_m + B / 4 * (cos_sigma * (-1 + 2 * cos2_sigma_m ** 2) -
                                    B / 6 * cos2_sigma_m * (-3 + 4 * sin_sigma ** 2) *
                                    (-3 + 4 * cos2_sigma_m ** 2)))

    distance = b * A * (sigma - delta_sigma) * 0.000621371  # Distance in miles

    return Decimal(distance)


def get_usage():
    """
        purpose: Reads daily usage from usage table
        input:
            no_of_days--> int
            usage_table--> str
            acct_type--> str
        output: spark dataframe

    """
    spark.sql(f"refresh table glue_catalog.usage_billing_dm.ami_register_read_daily_usage")
    usage = spark.sql(f"""
                                    SELECT CONTRACT_ACCOUNT_ID AS ACCOUNT_ID
                                            , COMPANY_CODE AS OPCO
                                            , METER_NUMBER
                                            , USAGE*MULTIPLIER AS USAGE
                                            , METER_READ_TIME AS READ_TIME
                                    FROM glue_catalog.usage_billing_dm.ami_register_read_daily_usage
                                    WHERE ACCOUNT_DETERMINATION_CODE = 'R'
                                        AND METER_NUMBER in (SELECT distinct METER_NUMBER
                                                            FROM glue_catalog.usage_billing_dm.ami_register_read_daily_usage)  
                                        AND METER_READ_TIME between current_date() - 750 and current_date()
                                    ORDER BY METER_NUMBER, METER_READ_TIME
                        """)
    return usage


# no_of_days = 366
# usage_table = 'glue_catalog.usage_billing_dm.ami_register_read_daily_usage'
# acct_type = 'R'
# mtrs=mtr_list.meters
# mtrs=['AM10012015']
usage = get_usage()
usage.createOrReplaceTempView("usage")


# get meter lat-long
def get_mtr_lat_long():
    """
    purpose: get meter latitude and longitude to calculate distance from airports
    input:
        customer_table-->str
        acct_type-->str
        mtrs-->list
    output:  spark dataframe
    """
    spark.sql(f"refresh table glue_catalog.customerdatawarehouse.ct_customer")
    mtr_lat_long = spark.sql(f"""
                        SELECT DISTINCT C.CONTRACT_ACCOUNT_ID AS ACCOUNT_ID
                                , C.MATERIAL_SERIAL_NUMBER AS METER_NUMBER
                                , C.COMPANY_CODE
                                , C.SERVICE_LOCATION_LATITUDE AS LAT
                                , C.SERVICE_LOCATION_LONGITUDE AS LONG
                        FROM glue_catalog.customerdatawarehouse.ct_customer C
                        WHERE C.MATERIAL_SERIAL_NUMBER in (SELECT distinct METER_NUMBER
                                                            FROM glue_catalog.usage_billing_dm.ami_register_read_daily_usage)  
                        """)
    return mtr_lat_long


mtr_lat_long = get_mtr_lat_long()
mtr_lat_long.createOrReplaceTempView("mtr_lat_long")


# create distance calculations
def get_airport_lat_long(airport_table, airport_list):
    """
        purpose: get airport's latitude and longitude to calculate distance from meter
        input:
            airport_table --> str
            airport_list --> list
        output:
            spark dataframe
    """
    spark.sql(f"refresh table {airport_table}")
    airport_lat_long = spark.sql(f"""
                    SELECT 
                        SITE_ID, 
                        LATITUDE AS LAT, 
                        LONGITUDE AS LONG
                    FROM {airport_table}
                    WHERE SITE_ID IN ('{"','".join(airport_list)}')
                    """)
    return airport_lat_long


airport_table = 'glue_catalog.weather_dq.wsi_site'
airport_list = ['KLIT', 'KBTR', 'KLCH', 'KMSY', 'KMLU', 'KJAN', 'KIAH', 'KBPT', 'KMEM']

airport_lat_long = get_airport_lat_long(airport_table, airport_list)
airport_lat_long.createOrReplaceTempView("airport_lat_long")


def get_nearest_airport(mtr_lat_long, airport_lat_long):
    """
        purpose:
        input:
                mtr_lat_long --> str
                airport_lat_long --> str
        output: spark dataframe
    """
    _ = spark.sql(f"""
                                SELECT DISTINCT 
                                    A1.ACCOUNT_ID,
                                    A1.METER_NUMBER,
                                    A2.SITE_ID AS AIRPORT,
                                    cast(A1.LAT as decimal(16, 10)) AS LAT1,
                                    cast(A1.LONG as decimal(16, 10)) AS LONG1,
                                    cast(A2.LAT as decimal(16, 10)) AS LAT2,
                                    cast(A2.LONG as decimal(16, 10)) AS LONG2
                                FROM mtr_lat_long A1
                                CROSS JOIN airport_lat_long A2
                                WHERE A1.LAT IS NOT NULL
                                    AND A1.LONG IS NOT NULL
                                    AND A2.LAT IS NOT NULL
                                    AND A2.LONG IS NOT NULL
                                    AND A1.LAT NOT LIKE ''
                                    AND A1.LONG NOT LIKE ''
                                group by a1.ACCOUNT_ID, A1.METER_NUMBER, A2.SITE_ID
                                            , A1.LAT, A1.LONG, A2.LAT, A2.LONG
                        """)

    distance_airport_meters = (
        _.withColumn("distance", calculate_distance(F.col("LAT1"), F.col("LONG1"), F.col("LAT2"), F.col("LONG2")))
        .drop("LAT1", "LONG1", "LAT2", "LONG2"))
    # get nearest airport
    distance_airport_meters.createOrReplaceTempView("distance_airport_meters")
    nearest_airport = spark.sql("""
                                    select na1.account_id, na1.meter_number, na1.airport, na2.distance
                                    from distance_airport_meters na1
                                    join (
                                        select
                                            account_id,
                                            meter_number,
                                            min(distance) as distance
                                            from distance_airport_meters
                                            group by account_id, meter_number
                                        ) na2
                                    on na1.account_id = na2.account_id
                                        and na1.meter_number = na2.meter_number
                                        and na1.distance = na2.distance
                                    order by account_id, meter_number
                                """)

    return nearest_airport


mtr_lat_long = mtr_lat_long
airport_lat_long = airport_lat_long

nearest_airport = get_nearest_airport(mtr_lat_long=mtr_lat_long, airport_lat_long=airport_lat_long)
nearest_airport.createOrReplaceTempView("nearest_airport")


def get_airport_weather(weather_table, no_of_days, airport_list):
    spark.sql(f"refresh table {weather_table}")
    hourly_temps = spark.sql(f"""select distinct
                                            t1.site_id as airport,
                                            to_date(t1.date_hr_lwt) as date ,
                                            hour(t1.date_hr_lwt) as hour,
                                            t1.date_hr_lwt as date_hour,
                                            t1.surf_temp_fht as temp
                                        from {weather_table} as t1
                                            where 1=1
                                                AND t1.site_id in ('KLIT', 'KBTR', 'KLCH', 'KMSY', 'KMLU', 'KJAN', 'KIAH', 'KBPT', 'KMEM')
                                                AND t1.date_hr_lwt >= current_date() - 750
                            """)
    print(hourly_temps.show())
    hourly_temps.createOrReplaceTempView("hourly_temps")

    airprt_dt_cnt = spark.sql("""select
                                        a.airport,
                                        a.date,
                                        count(date_hour) as hour_cnt
                                   from hourly_temps as a
                                        group by a.airport, a.date
                            """)
    airprt_dt_cnt.createOrReplaceTempView("airprt_dt_cnt")

    hourly_temps = spark.sql("""
                select
                        a.*,
                        b.hour_cnt
                   from hourly_temps as a 
                       join airprt_dt_cnt as b 
                           on a.airport = b.airport and a.date = b.date
                """)

    hourly_temps.createOrReplaceTempView("hourly_temps")

    airport_weather_temps = spark.sql("""
                        SELECT 
                            AIRPORT, 
                            DATE,
                            HOUR,
                            TEMP,
                            CASE 
                                WHEN AIRPORT IN ('KLIT', 'KMSY', 'KJAN', 'KIAH', 'KBPT', 'KMEM') AND TEMP < 60 THEN ROUND((60-TEMP)/HOUR_CNT,2) 
                                WHEN AIRPORT IN ('KBTR', 'KLCH', 'KMLU') AND TEMP < 65 THEN ROUND((65-TEMP)/HOUR_CNT,2)
                                ELSE 0
                            END AS CDD,
                            CASE 
                                WHEN AIRPORT IN ('KLIT', 'KMLU') AND TEMP > 70 THEN ROUND((TEMP-70)/HOUR_CNT,2) 
                                WHEN AIRPORT IN ('KBTR', 'KLCH', 'KMSY', 'KJAN', 'KIAH', 'KBPT', 'KMEM') AND TEMP > 65 THEN ROUND((TEMP-65)/HOUR_CNT,2)
                                ELSE 0
                            END AS HDD
                        FROM hourly_temps
                        """)
    del hourly_temps
    spark.catalog.dropTempView("hourly_temps")

    airport_hdd_cdd = (airport_weather_temps.groupBy("airport", "date")
                       .pivot("hour")
                       .agg(F.first("HDD").alias("hdd_hour"), F.first("CDD").alias("cdd_hour"))
                       .orderBy("airport", "date"))

    # rename cdd/hdd columns to fit original SAS format
    cols = airport_hdd_cdd.select(airport_hdd_cdd.colRegex("`^([0-9]+)_[ch]dd_hour$`")).columns
    for col in cols:
        num = re.match("[0-9]+", col).group(0)
        type = re.search("[ch].*", col).group(0).upper()
        airport_hdd_cdd = airport_hdd_cdd.withColumnRenamed(col, "{}_{}".format(type, num))

    return airport_hdd_cdd


weather_table = 'glue_catalog.weather_dq.wsi_historical_cleaned'
no_of_days = 366
airport_list = ['KLIT', 'KBTR', 'KLCH', 'KMSY', 'KMLU', 'KJAN', 'KIAH', 'KBPT', 'KMEM']

airport_hdd_cdd = get_airport_weather(weather_table, no_of_days, airport_list)
airport_hdd_cdd.createOrReplaceTempView("airport_hdd_cdd")


def prepare_training_data(usage, nearest_airport, airport_hdd_cdd, covid_start_date='2020-03-15'):
    # vars for data_to_forecast query
    this_year = datetime.today().year
    last_year = datetime.today().year - 1
    holiday_list = [date.strftime("%Y-%m-%d") for date, day in
                    holidays.UnitedStates(years=[this_year, last_year]).items() if
                    day in ['Thanksgiving', 'Labor Day', 'Memorial Day']]

    training_data = spark.sql(f"""
                        select
                            m.meter_number,
                            m.account_id,
                            m.opco,
                            m.usage,
                            u.*,
                            case
                                when dayofweek(u.date) in (1, 7) then 1
                                else 0
                                end as weekend,
                            case
                                when u.date between '{last_year - 1}-12-25' and '{last_year}-01-01' then 1
                                when u.date between '{last_year}-12-25' and '{this_year}-01-01' then 1
                                when u.date in {tuple(holiday_list)} then 1
                                else 0
                                end as holiday,
                            case
                                when m.opco in ('EGS', 'ELL', 'ENO', 'ETI') and u.date between '{this_year}-08-15' and '{this_year}-11-15' then 1
                                else 0
                                end as hurricane,
                            case
                                when u.date > '{covid_start_date}' then 1
                                else 0
                                end as covid
                        from usage m
                        join nearest_airport n
                            on m.meter_number = n.meter_number
                        join airport_hdd_cdd u
                                on n.airport = u.airport and m.read_time = u.date
                        where 1=1
                        order by m.meter_number, u.date
                        """)

    del this_year, last_year, holiday_list, airport_hdd_cdd
    return training_data


covid_start_date = '2020-03-15'
training_data = prepare_training_data(usage=usage
                                      , nearest_airport=nearest_airport
                                      , airport_hdd_cdd=airport_hdd_cdd
                                      , covid_start_date=covid_start_date)

from datetime import datetime, timedelta


def train_test_split(data, no_of_days_in_test=30):
    train = data[data['date'] < data['date'].max() - timedelta(days=no_of_days_in_test)]
    test = data[data['date'] >= data['date'].max() - timedelta(days=no_of_days_in_test)]
    return train, test


data = training_data.toPandas()
# .sort_values(by=['meter_number','date'])
# training_cols = [e for e in data.columns if e not in ('meter_number', 'account_id', 'opco', 'airport', 'date','usage')]
# X=data[training_cols]
# y=data['usage']

train, test = train_test_split(data=data, no_of_days_in_test=30)


def get_model(df):
    training_cols = [e for e in df.columns if
                     e not in ('meter_number', 'account_id', 'opco', 'airport', 'date', 'usage')]
    model_df = pd.DataFrame(columns=[e for e in df.columns if e not in ('date', 'usage')])

    for mtr in list(df['meter_number'].unique()):

        data = df[df['meter_number'] == mtr].sort_values(by=['date'])
        data = data.dropna()
        _1 = data[['meter_number', 'account_id', 'opco', 'airport']].drop_duplicates().reset_index(drop=True)
        X = data[training_cols]
        y = data['usage']
        model = linear_model.LinearRegression()
        try:
            model.fit(X, y)
            print(f'SUCCESSFULLY completed training model for meter number : \t {0}'.format(mtr))
        except:
            print(f'ERROR while training model for meter number : \t {0}'.format(mtr))
        # model=train_model(X,y,mtr)
        _2 = pd.DataFrame(columns=X.columns.to_list(), data=[model.coef_]).reset_index(drop=True)
        _3 = pd.DataFrame(columns=['intercept'], data=[model.intercept_]).reset_index(drop=True)
        mtr_model = pd.concat([_1, _2, _3], axis=1)
        model_df = pd.concat([model_df, mtr_model])
    return model_df.reset_index(drop=True)


model_ = get_model(df=train)
print("Model: \n", model_)
print("Model shape: ", model_.shape)
model_df = spark.createDataFrame(model_)
model_df.coalesce(1).write.option("header", True).csv(model_store_path)

from sklearn.metrics import r2_score


def model_eval(train, test):
    train = train.dropna()
    test = test.dropna()
    model_df = pd.DataFrame()
    cols = [e for e in train.columns if e not in ('meter_number', 'account_id', 'opco', 'airport', 'date', 'usage')]
    for mtr in list(train['meter_number'].unique()):
        m_df = pd.DataFrame()

        X_train = train[train['meter_number'] == mtr][cols]
        X_test = test[test['meter_number'] == mtr][cols]
        y_train = train[train['meter_number'] == mtr]['usage']
        y_test = test[test['meter_number'] == mtr]['usage']

        model = linear_model.LinearRegression()

        try:
            regr.fit(X, y)
        except:
            print(f'ERROR while training model for meter number : \t {mtr}')

        model.fit(X_train, y_train)

        r2 = r2_score(y_test, model.predict(X_test))
        print(mtr, r2)

        m_df['intercept'] = model.intercept_
        m_df[cols] = model.coef_
        m_df['r2'] = r2

        model_df = pd.concat([model_df, ])
    return model_df


model_ = model_eval(train, test)


def predict(data, model):
    data = data.dropna()
    pred_usage = pd.DataFrame()
    cols = [e for e in train.columns if e not in ('meter_number', 'account_id', 'opco', 'airport', 'date', 'usage')]
    for mtr in list(data['meter_number'].unique()):
        p = {}
        df = data[data['meter_number'] == mtr][cols]
        p['forecasted_usage'] = [model.predict(df)]
        p['meter_number'] = [mtr]
        pred_usage = pd.concat([pred_usage, pd.DataFrame(p)])
    return pred_usage.reset_index(drop=True)