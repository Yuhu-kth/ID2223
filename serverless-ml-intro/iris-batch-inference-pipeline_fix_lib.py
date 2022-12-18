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
        # is_run = True
        # while is_run: 
        #    is_run = 
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
    # ss_time = '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))
    # ss_time = '_'

    project = hopsworks.login()
    fs = project.get_feature_store()
    version_modi = 14
    mr = project.get_model_registry()
    model = mr.get_model("iris_modal", version=version_modi)
    model_dir = model.download()
    model = joblib.load(model_dir + "/iris_model.pkl")
    version_modi = 1
    feature_view = fs.get_feature_view(name="iris_modal", version=version_modi)
    batch_data = feature_view.get_batch_data()
    
    y_pred = model.predict(batch_data)
    offset = 2
    flower = y_pred[y_pred.size-offset]
    flower_url = "https://raw.githubusercontent.com/featurestoreorg/serverless-ml-course/main/src/01-module/assets/" + flower + ".png"
    print("Flower predicted: " + flower)
    img = Image.open(requests.get(flower_url, stream=True).raw)            
    img.save("./latest_iris_hann.png")
    dataset_api = project.get_dataset_api()    
    dataset_api.upload("./latest_iris_hann.png", "Resources/images", overwrite=True)
   
    iris_fg = fs.get_feature_group(name="iris_modal", version=version_modi)
    df = iris_fg.read() 
    #print(df)
    label = df.iloc[-offset]["variety"]
    label_url = "https://raw.githubusercontent.com/featurestoreorg/serverless-ml-course/main/src/01-module/assets/" + label + ".png"
    print("Flower actual: " + label)
    img = Image.open(requests.get(label_url, stream=True).raw)            
    img.save("./actual_iris_hann.png")
    dataset_api.upload("./actual_iris_hann.png", "Resources/images", overwrite=True)
    
    monitor_fg = fs.get_or_create_feature_group(name="iris_predictions",
                                                version=version_modi,
                                                primary_key=["datetime"],
                                                description="Iris flower Prediction/Outcome Monitoring"
                                                )
    
    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    data = {
        'prediction': [flower],
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
    dfi.export(df_recent, './df_recent_hann.png', table_conversion = 'matplotlib')
    dataset_api.upload("./df_recent_hann.png", "Resources/images", overwrite=True)
    
    predictions = history_df[['prediction']]
    labels = history_df[['label']]

    # Only create the confusion matrix when our iris_predictions feature group has examples of all 3 iris flowers
    print("Number of different flower predictions to date: " + str(predictions.value_counts().count()))
    if predictions.value_counts().count() == 3:
        results = confusion_matrix(labels, predictions)
    
        df_cm = pd.DataFrame(results, ['True Setosa', 'True Versicolor', 'True Virginica'],
                             ['Pred Setosa', 'Pred Versicolor', 'Pred Virginica'])
    
        cm = sns.heatmap(df_cm, annot=True)
        fig = cm.get_figure()
        fig.savefig("./confusion_matrix_hann.png")
        dataset_api.upload("./confusion_matrix_hann.png", "Resources/images", overwrite=True)
        # return False
    else:
        print("You need 3 different flower predictions to create the confusion matrix.")
        print("Run the batch inference pipeline more times until you get 3 different iris flower predictions") 
        # return True

if __name__ == "__main__":
    if LOCAL == True :
        # is_run = True
        # while is_run: 
        #    is_run = 
        g()
    else:
        with stub.run():
            f()

