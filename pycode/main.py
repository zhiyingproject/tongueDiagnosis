import sys
import numpy as np
import tensorflow as tf
from keras.utils import np_utils
import os
from tensorflow import keras
import tables
sys.path.append('./myClass/')
import preprocessing
import create_model



categories = ['spike', 'color', 'shape', 'crack', 'coatingColor', 'coatingAmount']
original_path = "../originalData/apparatus"
config_path = "../config"
model_name = "iteration1"
epoch = 16
img_height = 166
img_width = 128
input_shape = (img_height, img_width, 3)

for category in categories:
    print('================================')
    print(f'Predicting {category}')
    print('================================')


    checkpoint_path = f'../models/{category}/{model_name}/cp-{epoch}.ckpt'
    model_path = f'../models/{category}/{model_name}'
    checkpoint_dir = os.path.dirname(checkpoint_path)
    model = keras.models.load_model(model_path)

    hdf5_path = f"../datasets/hdf5{category}"
    hdf5_file = tables.open_file(hdf5_path, mode='r')
    labels = preprocessing.preprocessing(category, config_path, original_path,
                                         hdf5_file, img_width, img_height).get_label_no()
    num_classes = len(list(labels.keys()))
    print(labels)
#    model_name = f"{category}.model.weights.best.hdf5"
    # model.load_weights(f'{category}.model.weights.best.hdf5')
    test_data = np.array(hdf5_file.root.test_img)
    test_label_tmp = np.array(hdf5_file.root.test_labels)
    test_label = np_utils.to_categorical(test_label_tmp, num_classes)
    score = model.evaluate(test_data, test_label, verbose=0)

    print('\n', 'Test accuracy:', score[1])
    classes = list(labels.values())
    for i, d in enumerate(test_data):
        d_e = (np.expand_dims(d, 0))

        predictions = model.predict(d_e)
        score = tf.nn.softmax(predictions[0])
        print(score)
        print(100 * np.max(score), classes[np.argmax(score)], classes[int(test_label_tmp[i])])
