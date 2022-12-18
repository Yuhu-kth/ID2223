import gradio as gr
import numpy as np
from PIL import Image
import requests

import hopsworks
import joblib

project = hopsworks.login()
fs = project.get_feature_store()


mr = project.get_model_registry()
model = mr.get_model("titanic_modal", version=2)
model_dir = model.download()
model = joblib.load(model_dir + "/titanic_model.pkl")


def titanic(Pclass,Sex,Age,SibSp,Parch,Fare,Embarked):
    input_list = []
    input_list.append(Pclass)
    input_list.append(Sex)
    input_list.append(Age)
    input_list.append(SibSp)
    input_list.append(Parch)
    input_list.append(Fare)
    input_list.append(Embarked)
    # 'res' is a list of predictions returned as the label.
    res = model.predict(np.asarray(input_list).reshape(1, -1)) 
    alive_url = "https://illustoon.com/photo/dl/1045.png"
    died_url = "https://illustoon.com/photo/dl/1052.png"
    img_path = alive_url if int(res[0]) == 1 else died_url  
    img = Image.open(requests.get(img_path, stream=True).raw)            
    return img
        
demo = gr.Interface(
    fn=titanic,
    title="Titanic Survival Predictive Analytics",
    description="Experiment with passengers information to predict whether they can survive in titanic.",
    allow_flagging="never",
    inputs=[
        gr.inputs.Number(default=1.0, label="Class [0, 1, 2]"),
        gr.inputs.Number(default=1.0, label="Sex [0(male), 1(female)]"),
        gr.inputs.Number(default=1.0, label="Age [y/o]"),
        gr.inputs.Number(default=1.0, label="sibsp [0-5]]"),
        gr.inputs.Number(default=1.0, label="Parch [0-6]]"),
        gr.inputs.Number(default=1.0, label="Fare [USD]"),
        gr.inputs.Number(default=1.0, label="Embarked [0 (S), 1 (C),  2 (Q)]"),
        ],
    outputs=gr.Image(type="pil"))

demo.launch()

