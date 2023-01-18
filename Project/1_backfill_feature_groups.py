import pandas as pd
from functions import *
import numpy as np
# Load air quality data
df_air_quality = pd.read_csv('beijing-air-quality.csv')
print(df_air_quality.head())

df_air_quality = add_city_column(df_air_quality, 'Beijing')
df_air_quality.rename(
    columns={" pm25": "pm25", " pm10": "pm10"," o3": "o3"," no2":"no2"," so2":"so2"," co":"co"}, inplace=True)
df_air_quality.date = df_air_quality.date.apply(timestamp_2_time)
df_air_quality.sort_values(by = ['date'], inplace = True, ignore_index = True)


# Load weather data
df_weather = pd.read_csv('beijing-weather.csv')
print(df_weather.head())

df_weather.rename(
    columns={"name": "city", "datetime": "date"}, inplace=True)

df_weather.date = df_weather.date.apply(timestamp_2_time_weather)
df_weather.sort_values(by=['date'], inplace=True, ignore_index=True)

print(df_weather.head())

#Replace NaNs with 0
df_air_quality = remove_nans_in_csv(df_air_quality)
df_weather = remove_nans_in_csv(df_weather)

#Remove features that we don't get from the API
df_weather = df_weather.drop(columns=["precipprob", "preciptype", "uvindex",
                             "severerisk", "sunrise", "sunset", "moonphase", "description", "icon", "stations"])
print(df_air_quality.shape)
df_air_quality = df_air_quality.drop(columns=["no2", "so2","co"])
df_air_quality = df_air_quality.replace(r'^\s+$', np.nan, regex=True)
df_air_quality = df_air_quality.replace(np.nan,0,regex = True)
df_air_quality['pm25'] = df_air_quality['pm25'].astype(float)
df_air_quality['pm10'] = df_air_quality['pm10'].astype(float)
df_air_quality['o3'] = df_air_quality['o3'].astype(float)
print("DF AIR ---------------------------------------------")
# print(df_air_quality.shape)
print(df_air_quality.head())
# print("DF WEATHER -----------------------------------------")
# print(df_weather.shape)
# print(df_weather.head())

#Connect to Hopsworks, create and upload feature groups
import hopsworks
project = hopsworks.login()
fs = project.get_feature_store() 

air_quality_fg = fs.get_or_create_feature_group(
        name = 'air_quality_fg',
        description = 'Air Quality characteristics of each day',
        version = 3,
        primary_key = ['city','date'],
        online_enabled = True,
        event_time = 'date'
    )    
air_quality_fg.insert(df_air_quality)

weather_fg = fs.get_or_create_feature_group(
        name = 'weather_fg',
        description = 'Weather characteristics of each day',
        version = 1,
        primary_key = ['city','date'],
        online_enabled = True,
        event_time = 'date'
    )
weather_fg.insert(df_weather)