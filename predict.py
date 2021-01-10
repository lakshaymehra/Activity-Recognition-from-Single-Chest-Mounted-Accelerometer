# Run file using: !python predict.py --model_path 'C:\Users\mehra\PycharmProjects\VicaraAssignment\knn_model.pkl' --model_name 'knn' --x_acc 2145 --y_acc 2336 --z_acc 1947

import numpy as np
import argparse
import pickle
import keras

def knn_xg_predict(model_path, x_acc, y_acc, z_acc):

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    prediction = model.predict(np.expand_dims(np.array([x_acc, y_acc, z_acc]), axis=0))
    predicted_class_id = prediction[0]
    return predicted_class_id

def nn_predict(model_path, x_acc, y_acc, z_acc):

    new_model = keras.models.load_model(model_path)
    prediction = new_model.predict(np.expand_dims(np.array([x_acc, y_acc, z_acc]), axis=0))
    predicted_class_id = np.argmax(prediction, axis=1)[0]
    return predicted_class_id

def get_label(predicted_class_id):

    labels = ['No Activity', 'Working at Computer', 'Standing Up, Walking and Going up\down stairs', 'Standing',
              'Walking', 'Going Up\Down Stairs', 'Walking and Talking with Someone', 'Talking while Standing']
    predicted_label = labels[predicted_class_id]
    return predicted_label

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        help='Specify the path to the model file.')
    parser.add_argument(
        '--model_name',
        type=str,
        help='Specify the model name.',
        choices=['knn', 'xgboost', 'neural_network'])
    parser.add_argument(
        '--x_acc',
        type=int,
        help='Specify the X-Acceleration value.')
    parser.add_argument(
        '--y_acc',
        type=int,
        help='Specify the Y-Acceleration value.')
    parser.add_argument(
        '--z_acc',
        type=int,
        help='Specify the Z-Acceleration value.')

    args = parser.parse_args()
    model_path = args.model_path
    model_name = args.model_name
    x_acc = args.x_acc
    y_acc = args.y_acc
    z_acc = args.z_acc

    if model_name == 'neural_network':
        pred_id = nn_predict(model_path,x_acc,y_acc,z_acc)

    else:
        pred_id = knn_xg_predict(model_path, x_acc, y_acc, z_acc)

    pred_label = get_label(pred_id)
    print('Predicted Activity: ', pred_label)