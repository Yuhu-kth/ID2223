# ID2223-Scalable-ml-and-dl
# Project- Beijing Air quality prediction
### This project is about predicting the future air quality of the next seven days in Beijing.
### The online interface of this project can be accessed via huggingface: https://huggingface.co/spaces/Hannnnnah/Air_quality 
### For now, it only supports the input of "Beijing", the output is "pm10" of this location in the next seven days.
## Data collection
### Data for air quality was downloaded from the World Air Quality Project: https://aqicn.org//here/ ,
### Data for weather conditions was downloaded from Visula Crossing: https://www.visualcrossing.com/ .
## Steps to run this project
### 1.back_feature_groups.py : download historical air quality data and weather data from the Air Quality Project and Visual Crossing; data processing and create feature groups onto hopsworks.
### 2.feature_pipline.py: download daily data and upload to feature group.
### 3.feature_views_and_training_dataset.ipynb : combine air_quality feature group with weather group and then create feature view, then create dataset.
### 4.model_training_and_prediction.ipynb : Train the model using Gradient Boosting Regressor, the metrics id MAE(mean absolute error)
### 5.app.py is the code for the demo on huggingface
### 6.functions.py consists the functions used in the files above.

