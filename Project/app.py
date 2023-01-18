import gradio as gr
from datetime import timedelta, datetime
import hopsworks
import joblib
from functions import *
os.environ['AIR_QUALITY_API_KEY'] = '2af710d5722e03e360f0705e26797ca3027a7bbe'
os.environ['WEATHER_API_KEY'] =  'X3347MZK4ULFFF3PK4MRLVKKY'
#Connect to hopsworks and get feature store
project = hopsworks.login()
fs = project.get_feature_store()

#Function for the app
def predict_weather(location):

    #Get future weather data
    weather_data1 = get_weather_df([get_weather_data(location, (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"))])
    weather_data2 = get_weather_df([get_weather_data(location, (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d"))])
    weather_data3 = get_weather_df([get_weather_data(location, (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d"))])
    weather_data4 = get_weather_df([get_weather_data(location, (datetime.now() + timedelta(days=4)).strftime("%Y-%m-%d"))])
    weather_data5 = get_weather_df([get_weather_data(location, (datetime.now() + timedelta(days=5)).strftime("%Y-%m-%d"))])
    weather_data6 = get_weather_df([get_weather_data(location, (datetime.now() + timedelta(days=6)).strftime("%Y-%m-%d"))])
    weather_data7 = get_weather_df([get_weather_data(location, (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d"))])

    weather_df = pd.concat([weather_data1, weather_data2, weather_data3, weather_data4, weather_data5, weather_data6, weather_data7], axis=0)

    weather_df = weather_df.drop(columns=["precipprob", "uvindex", "date", "city", "conditions"]).fillna(0)
    weather_df.rename(
        columns={"pressure": "sealevelpressure"}, inplace=True)
    print(weather_data1)

    #Get model
    mr = project.get_model_registry()
    model = mr.get_model("Gradient_Duster", version=1)
    model_dir = model.download()
    model = joblib.load(model_dir + "/Gradient_Duster.pkl")
    print("model")
    #Create predictions
    preds = model.predict(weather_df)
    print(preds)

    list_of_predictions = []
    for x in range(7):
      list_of_predictions.append("pm10 on " + (datetime.now() + timedelta(days=x+1)).strftime('%Y-%m-%d') + ": " +  str(int(preds[x])))

    return list_of_predictions

#Gradio interface
demo = gr.Interface(
    fn=predict_weather,
    title="Future air quality predictor",
    description="Input the name of a location below to get future air quality predictions for that location",
    allow_flagging="never",
    inputs="text",
    outputs="text"
)

demo.launch(share=True)