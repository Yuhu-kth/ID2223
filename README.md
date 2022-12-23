# ID2223-Scalable-ml-and-dl
## Lab1
### task1 Iris Dataset
#### Huggingface Iris : https://huggingface.co/spaces/Hannnnnah/Iris
#### Huggingface Iris monitor : https://huggingface.co/spaces/Hannnnnah/Iris_monitor
### task2 Titanic Dataset
#### Huggingface Titanic : https://huggingface.co/spaces/Hannnnnah/Titanic
#### Huggingface Titanic monitor : https://huggingface.co/spaces/Hannnnnah/Titanic_monitor

## Lab2 Whisper
#### Huggingface whisper: https://huggingface.co/spaces/Hannnnnah/Whisper-small
#### Ways that can improve model's performance:
#### Model-Centric:
#### Parameters setting:
#### 1. Increase the number of epoch, which means that it takes more time to train. It helps to fit the data better but it may cause overfitting. The computation consumption is also considered.
#### 2. Adjust the learning rate.Higher learning rate may converge fast but with high risk of missing local maxima.Smaller learning rate may take long time to update the weights. 
#### 3. Adjust the batch size. If you feed more data for each step to the model, it takes longer to update.
#### Model choice:
#### In this experiment, we used Whisper-small model, there are other models can be used , eg: Whisper-large. However, it may take longer time than we expected.

#### Data-Centric:
#### 1. Include more data
#### 2. Data augmentation: Process the data with different methods like cutting each sample into several clips and then shuffle them and re-combine them together. This can be done under the situation of limited data.


