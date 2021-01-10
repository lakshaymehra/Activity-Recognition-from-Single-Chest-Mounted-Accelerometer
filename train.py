# Run using !python train.py --directory_path 'C:/Users/mehra/PycharmProjects/VicaraAssignment/Activity Recognition from Single Chest-Mounted Accelerometer/Activity Recognition from Single Chest-Mounted Accelerometer' --model_name 'knn'

import numpy as np
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as knc
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import xgboost as xgb
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import pickle

def find_neighbors(x_train,y_train,x_test,y_test):

    accur = []
    optimum = 0
    val3 = 0
    for i in range(1, 25):
        k = i+1
        neighbors = knc(n_neighbors = k)
        neighbors.fit(x_train, y_train)
        y_predict = neighbors.predict(x_test)
        a = metrics.accuracy_score(y_test, y_predict)*100
        accur.append(a)
        if a > val3:
            val3 = a
            optimum = k
        print("Accuracy is ", a, "% for k value : ", k)

    print("\n The optimum number of neighbors for this dataset is ", optimum)
    plt.figure(figsize=(16,5))
    plt.title("Model Accuracy Score: \n")
    plt.ylabel("Accuracy Scores: (in %)")
    plt.ylim(40, 100)
    plt.xlim(0, 25)
    plt.xlabel("Neighbors")
    plt.plot(range(1, 25), accur)
    plt.show()
    return optimum

def train_knn(x_train,y_train,x_test,y_test,optimum):

    knn = knc(optimum)
    knn_model = knn.fit(x_train, y_train)
    y_train_pred = knn_model.predict(x_train)
    y_pred = knn_model.predict(x_test)
    # print(y_pred)
    # print(y_test)
    print("KNN Accuracy Score for Train Set: ", metrics.accuracy_score(y_train, y_train_pred))
    print("KNN Accuracy Score for Test Set: ", metrics.accuracy_score(y_test, y_pred))
    print("KNN Classification Report: \n", metrics.classification_report(y_test, y_pred))
    with open('knn_model.pkl', 'wb') as f:
        pickle.dump(knn_model, f)
    cm = metrics.confusion_matrix(y_test, y_pred)
    print("KNN Confusion Matrix:\n", cm)
    return cm

def train_xgboost(x_train,y_train,x_test,y_test):

    model = xgb.XGBClassifier(random_state=1, learning_rate=0.01)
    xg_model = model.fit(x_train, y_train)
    y_train_pred = xg_model.predict(x_train)
    y_pred = xg_model.predict(x_test)
    # print(y_pred)
    # print(y_test)
    print("XGBoost Accuracy Score for Train Set: ", metrics.accuracy_score(y_train, y_train_pred))
    print("XGBoost Accuracy Score for Test Set: ", metrics.accuracy_score(y_test, y_pred))
    print("XGBoost Classification Report: \n", metrics.classification_report(y_test, y_pred))
    with open('xg_model.pkl', 'wb') as f:
        pickle.dump(xg_model, f)
    cm = metrics.confusion_matrix(y_test, y_pred)
    print("XGBoost Confusion Matrix:\n", cm)
    return cm

def train_nn(x_train,y_train,x_test,y_test):

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    # print(y_train.shape)

    model = Sequential()
    model.add(Dense(1024, input_shape=(3,), activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(8, activation='softmax'))

    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fit the model
    model.fit(x_train, y_train, epochs=5)
    model.save('neural_network_model.h5')
    y_nn_pred = model.predict(x_test)
    # print(y_nn_pred)
    y_nn_pred = np.argmax(y_nn_pred, axis=1)
    # print(y_nn_pred)

    y_test = [np.argmax(y, axis=None, out=None) for y in y_test]
    # print(y_test)

    print("Neural Network Accuracy Score for Test Set: ",metrics.accuracy_score(y_test, y_nn_pred))
    print("Neural Network Classification Report: \n", metrics.classification_report(y_test, y_nn_pred))
    cm = metrics.confusion_matrix(y_test, y_nn_pred)
    print("Neural Network Confusion Matrix:\n", cm)
    return cm

def visualize_confusion_matrix(cm):

    print("\nVisualising confusion matrix : \n")
    fig = plt.figure(figsize = (27, 5))
    ax= plt.subplot()
    target_names = ['No Activity', 'Working at Computer', 'Standing Up, Walking and Going up\down stairs', 'Standing',
              'Walking', 'Going Up\Down Stairs', 'Walking and Talking with Someone', 'Talking while Standing']
    sns.set(font_scale=1.4)
    sns.heatmap(cm, annot=True, ax = ax, fmt='g')
    ax.set_xlabel('Predicted labels', fontweight="bold")
    ax.set_ylabel('True labels', fontweight="bold")
    ax.set_title('Confusion Matrix', fontweight="bold")
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    ax.xaxis.set_ticklabels(target_names); ax.yaxis.set_ticklabels(target_names)
    plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--directory_path',
        type=str,
        help='Specify the path to the directory containing the data.')
    parser.add_argument(
        '--model_name',
        type=str,
        help='Specify the model name.',
        choices=['knn','xgboost','neural_network'])

    args = parser.parse_args()
    directory_path = args.directory_path
    model_name = args.model_name

    dir = glob(directory_path + "/*.csv")
    df = pd.DataFrame()

    for i, name in enumerate(dir):
        temp_df = pd.read_csv(name, header=None)
        df = df.append(temp_df)

    del df[0]
    df.columns = ['X-acceleration', 'Y-acceleration', 'Z-acceleration', 'Activity_ID']
    # print(df)
    # print(df['Activity_ID'].value_counts())

    x = df.iloc[:, 0:3]
    y = df.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

    if model_name == 'knn':
        optimum = find_neighbors(x_train, y_train, x_test, y_test)
        cm = train_knn(x_train, y_train, x_test, y_test, optimum)
        visualize_confusion_matrix(cm)

    elif model_name == 'xgboost':
        cm = train_xgboost(x_train, y_train, x_test, y_test)
        visualize_confusion_matrix(cm)

    else:
        cm = train_nn(x_train, y_train, x_test, y_test)
        visualize_confusion_matrix(cm)
