import pandas as pd
import numpy as np
from datetime import datetime
from functions import *
os.environ['AIR_QUALITY_API_KEY'] = '2af710d5722e03e360f0705e26797ca3027a7bbe'
os.environ['WEATHER_API_KEY'] =  'X3347MZK4ULFFF3PK4MRLVKKY'
date_today = datetime.now().strftime("%Y-%m-%d")
cities = ['Beijing']
data_air_quality = [get_air_quality_data("Beijing")]
data_weather = [get_weather_data("Beijing", date_today)]

df_air_quality = get_air_quality_df(data_air_quality)

df_air_quality.drop(['aqi','iaqi_h', 'iaqi_p', 'iaqi_pm10', 'iaqi_t', 'o3_max', 'o3_min', 'pm10_max', 'pm10_min', 'pm25_max', 'pm25_min', 'uvi_avg', 'uvi_max', 'uvi_min'], axis=1, inplace=True)
df_air_quality.rename(
    columns={"o3_avg": "o3", "pm10_avg": "pm10", "pm25_avg": "pm25"}, inplace=True)
df_air_quality = df_air_quality.replace(r'^\s+$', np.nan, regex=True)
df_air_quality = df_air_quality.replace(np.nan,0,regex = True)
df_air_quality['pm25'] = df_air_quality['pm25'].astype(float)
df_air_quality['pm10'] = df_air_quality['pm10'].astype(float)
df_air_quality['o3'] = df_air_quality['o3'].astype(float)

print(df_air_quality.columns)
print(df_air_quality)

df_weather = get_weather_df(data_weather)
df_weather = df_weather.drop(columns=["precipprob", "uvindex"])
df_weather.rename(
    columns={"pressure": "sealevelpressure"}, inplace=True)
print(df_weather.head())

#Connect to Hopsworks and upload data
import hopsworks
project = hopsworks.login()
fs = project.get_feature_store() 

air_quality_fg = fs.get_or_create_feature_group(
    name = 'air_quality_fg',
    primary_key = ['date'],
    version = 3
)
air_quality_fg.insert(df_air_quality)

weather_fg = fs.get_or_create_feature_group(
   name = 'weather_fg',
    primary_key = ['city', 'date'],
   version = 1
)
weather_fg.insert(df_weather)