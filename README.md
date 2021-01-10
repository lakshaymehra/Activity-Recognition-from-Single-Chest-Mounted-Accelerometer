# Vicara Activity Recognition Using Machine Learning Assignment :

## Usage Guide:

### File Descriptions:

* ##### 'exploratory_data_analysis.py' :
    contains the code to perform basic Exploratory Data Analysis(EDA) using various Visualizations.

* ##### 'train.py' :
    contains the code to train K-Nearest-Neighbors, XGBoost, and Neural Network Models.

* ##### 'predict.py' :
    contains the code to load the pre-trained models and predict Activity using 3-DOF sensor raw values (X-acceleration, Y-acceleration, and Z-acceleration) as input. 

* ##### 'knn_model.pkl' :
    contains the KNN model that I have trained. 

* ##### 'xg_model.pkl' :
    contains the XGBoost model that I have trained. 
  
* ##### 'neural_network_model.h5' :
    contains the Neural Network model that I have trained. 

* ##### 'requirements.txt' :
    contains the packages and their exact versions required to run the python scripts in this repository. 

### Running Files:

* ##### To perform EDA, use:
```bash
 python .\exploratory_data_analysis.py --directory_path 'C:\Users\mehra\PycharmProjects\VicaraAssignment\Activity Recognition from Single Chest-Mounted Accelerometer\Activity Recognition from Single Chest-Mounted Accelerometer'
```
Change the value of the directory_path to specify the directory where you have saved the data files.

* ##### To train models, use:
```bash
python train.py --directory_path 'C:/Users/mehra/PycharmProjects/VicaraAssignment/Activity Recognition from Single Chest-Mounted Accelerometer/Activity Recognition from Single Chest-Mounted Accelerometer' --model_name 'knn'
```
Change the value of the directory_path to specify the directory where you have saved the data files.
Change the value of the model name to train KNN, XGBoost, and NN according to your requirements.

* ##### To predict, use:
```bash
python predict.py --model_path C:\Users\mehra\PycharmProjects\VicaraAssignment\neural_network_model.h5 --model_name 'neural_network' --x_acc 2145 --y_acc 2336 --z_acc 1947
```
Change the value of the directory_path to specify the directory where you have saved the data files.
Change the value of the model name to train KNN, XGBoost, and NN according to your requirements.
Change the values of X,Y, and Z acceleration according to your requirements.

## Exploratory Data Analysis(EDA):

### Activity Counts:
![Alt text](Figure_5.png?raw=true "Figure_5")

### Cross-Tab:
![Alt text](Figure_6.png?raw=true "Figure_6")

### Acceleration Trends For Each Activity:
![Alt text](Figure_7.png?raw=true "Figure_7")
![Alt text](Figure_8.png?raw=true "Figure_8")
![Alt text](Figure_9.png?raw=true "Figure_9")
![Alt text](Figure_10.png?raw=true "Figure_10")
![Alt text](Figure_11.png?raw=true "Figure_11")
![Alt text](Figure_12.png?raw=true "Figure_12")
![Alt text](Figure_13.png?raw=true "Figure_13")
![Alt text](Figure_14.png?raw=true "Figure_14")

## Results:

### K-Nearest-Neighbors Model Results:

Accuracy is  65.03243551818983 % for k value :  2

Accuracy is  69.79033681042088 % for k value :  3

Accuracy is  71.14899579635684 % for k value :  4

Accuracy is  72.19264102963308 % for k value :  5

Accuracy is  72.68540142197311 % for k value :  6

Accuracy is  73.25730447869635 % for k value :  7

Accuracy is  73.53599045098345 % for k value :  8

Accuracy is  73.86657325237428 % for k value :  9

Accuracy is  73.99605584098812 % for k value :  10

Accuracy is  74.21168716591417 % for k value :  11

Accuracy is  74.30691784731953 % for k value :  12

Accuracy is  74.39851575068764 % for k value :  13

Accuracy is  74.45819710415694 % for k value :  14

Accuracy is  74.59286937568115 % for k value :  15

Accuracy is  74.65021537184077 % for k value :  16

Accuracy is  74.70107426436245 % for k value :  17

Accuracy is  74.73843998131714 % for k value :  18

Accuracy is  74.75971768124967 % for k value :  19

Accuracy is  74.79500752504022 % for k value :  20

Accuracy is  74.82458871762935 % for k value :  21

Accuracy is  74.85183455290881 % for k value :  22

Accuracy is  74.93720483678447 % for k value :  23

Accuracy is  74.92241424048991 % for k value :  24

Accuracy is  74.992993928071 % for k value :  25

The optimum number of neighbors for this dataset is  25

![Alt text](Figure_1.png?raw=true "Figure_1")

KNN Accuracy Score for Train Set:  0.7627958451290807

#### KNN Accuracy Score for Test Set:  0.7509030048264052

Classification Report: 

               precision    recall  f1-score   support

           0       0.41      0.03      0.06       811
           1       0.86      0.91      0.89    121138
           2       0.59      0.18      0.28      9720
           3       0.61      0.47      0.53     43571
           4       0.64      0.75      0.69     71630
           5       0.43      0.12      0.19     10103
           6       0.53      0.21      0.30      9578
           7       0.77      0.84      0.80    118829

    accuracy                           0.75    385380
    macro avg       0.60      0.44      0.47    385380
    weighted avg       0.74      0.75      0.73    385380

![Alt text](Figure_2.png?raw=true "Figure_2")


### XGBoost Model Results:

XGBoost Accuracy Score for Train Set:  0.678043562311387

#### XGBoost Accuracy Score for Test Set:  0.6795863822720432

XGBoost Classification Report:

               precision    recall  f1-score   support

           0       0.67      0.00      0.01       770
           1       0.76      0.88      0.82    121656
           2       0.63      0.05      0.10      9522
           3       0.52      0.29      0.37     43271
           4       0.64      0.61      0.62     70936
           5       0.27      0.00      0.00     10211
           6       0.43      0.05      0.08      9482
           7       0.64      0.82      0.72    119532

    accuracy                           0.68    385380
    macro avg       0.57      0.34      0.34    385380
    weighted avg       0.65      0.68      0.64    385380

### Neural Network Model Results:

Epoch 1/5
48173/48173 [==============================] - 123s 3ms/step - loss: 2.0535 - accuracy: 0.4762

Epoch 2/5
48173/48173 [==============================] - 122s 3ms/step - loss: 1.2062 - accuracy: 0.5730

Epoch 3/5
48173/48173 [==============================] - 123s 3ms/step - loss: 1.1845 - accuracy: 0.5816

Epoch 4/5
48173/48173 [==============================] - 122s 3ms/step - loss: 1.1672 - accuracy: 0.5899

Epoch 5/5
48173/48173 [==============================] - 123s 3ms/step - loss: 1.1525 - accuracy: 0.5959

#### Neural Network Accuracy Score for Test Set: 0.5619751933156885

### Thus, it is clearly evident that KNN outperforms XGBoost and Dense Neural Network for this dataset.
### Contact :
For any question, please contact
```
Lakshay Mehra: mehralakshay2@gmail.com
```

