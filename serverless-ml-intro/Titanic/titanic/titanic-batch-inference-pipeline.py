import os
import modal
import sklearn
# need lower than 1.0.2 (pip install scikit-learn==0.24.0)
LOCAL=False

if LOCAL == False:
   stub = modal.Stub()
   hopsworks_image = modal.Image.debian_slim().pip_install(["hopsworks==3.0.4","scikit-learn==0.24.0","joblib","seaborn","sklearn","dataframe-image"])
   import sklearn
   @stub.function(image=hopsworks_image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("my-custom-secret"))
   def f():
        print("finished env, into the g function")
        g()

def g():
    import pandas as pd
    import hopsworks
    import joblib
    import datetime
    from PIL import Image
    from datetime import datetime
    import dataframe_image as dfi
    from sklearn.metrics import confusion_matrix
    from matplotlib import pyplot
    import seaborn as sns
    import requests

    now = datetime.now()
    project = hopsworks.login()
    fs = project.get_feature_store()
    version_modi = 1
    mr = project.get_model_registry()
    model = mr.get_model("titanic_modal", version=version_modi)
    model_dir = model.download()
    model = joblib.load(model_dir + "/titanic_model.pkl")
    version_modi = 1
    feature_view = fs.get_feature_view(name="titanic_modal", version=version_modi)
    batch_data = feature_view.get_batch_data()
    
    y_pred = model.predict(batch_data)
    offset = 3
    survived = y_pred[y_pred.size-offset]
    alive_url = "https://illustoon.com/photo/dl/1045.png"
    died_url = "https://illustoon.com/photo/dl/1052.png"
    img_path = alive_url if int(survived) == 1 else died_url
    print("Passenger predicted:"+str(survived))
    img = Image.open(requests.get(img_path, stream=True).raw)            
    img.save("./latest_passenger.png")
    dataset_api = project.get_dataset_api()    
    dataset_api.upload("./latest_passenger.png", "Resources/images", overwrite=True)

    titanic_fg = fs.get_feature_group(name="titanic_modal", version=version_modi)
    df = titanic_fg.read() 
    label = df.iloc[-offset]["survived"]
    label_url = alive_url if int(label)==1 else died_url
    print("Survival actual: " + str(label))
    img = Image.open(requests.get(label_url, stream=True).raw)            
    img.save("./actual_passenger.png")
    dataset_api.upload("./actual_passenger.png", "Resources/images", overwrite=True)

    monitor_fg = fs.get_or_create_feature_group(name="titanic_predictions",
                                                version=version_modi,
                                                primary_key=["datetime"],
                                                description="Titanic survival Prediction/Outcome Monitoring"
                                                )
    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    data = {
        'prediction': [survived],
        'label': [label],
        'datetime': [now],
       }
    monitor_df = pd.DataFrame(data)
    monitor_fg.insert(monitor_df, write_options={"wait_for_job" : False})
    
    history_df = monitor_fg.read()
    # Add our prediction to the history, as the history_df won't have it - 
    # the insertion was done asynchronously, so it will take ~1 min to land on App
    history_df = pd.concat([history_df, monitor_df])

    df_recent = history_df.tail(4)
    dfi.export(df_recent, './df_recent_titanic.png', table_conversion = 'matplotlib')
    dataset_api.upload("./df_recent_titanic.png", "Resources/images", overwrite=True)
    
    predictions = history_df[['prediction']]
    labels = history_df[['label']]

    # Only create the confusion matrix when our iris_predictions feature group has examples of all 3 iris flowers
    print("Number of different survival predictions to date: " + str(predictions.value_counts().count()))
    if predictions.value_counts().count() == 2:
        results = confusion_matrix(labels, predictions)
    
        df_cm = pd.DataFrame(results, ['True Died', 'True Survived'],
                         ['Pred Died', 'Pred Survived'])
    
        cm = sns.heatmap(df_cm, annot=True)
        fig = cm.get_figure()
        fig.savefig("./confusion_matrix_titanic.png")
        dataset_api.upload("./confusion_matrix_titanic.png", "Resources/images", overwrite=True)
    else:
        print("You need 2 different survival predictions to create the confusion matrix.")
        print("Run the batch inference pipeline more times until you get 2 different iris flower predictions") 


if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()

