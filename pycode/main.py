import sys
import numpy as np
import tensorflow as tf
from keras.utils import np_utils
import os
from tensorflow import keras
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score, roc_auc_score, confusion_matrix
import tables
sys.path.insert(0, './myClass/')
import preprocessing
import create_model


categories = ['spike', 'color', 'shape', 'crack', 'coatingColor', 'coatingAmount']
original_path = "../originalData/apparatus"
config_path = "../config"
model_name = "iteration2"
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


    hdf5_path = f"../datasets/hdf5{category}"
    hdf5_file = tables.open_file(hdf5_path, mode='r')
    labels = preprocessing.hdf5File(category, config_path, original_path,
                                         hdf5_file, img_width, img_height).get_label_no()
    num_classes = len(list(labels.keys()))
    model = create_model.tf_models(model_name, input_shape, num_classes)
    model = keras.models.load_model(model_path)
    print(labels)
#    model_name = f"{category}.model.weights.best.hdf5"
    # model.load_weights(f'{category}.model.weights.best.hdf5')
    test_data = np.array(hdf5_file.root.test_img)
    test_label_tmp = np.array(hdf5_file.root.test_labels)
    test_label = np_utils.to_categorical(test_label_tmp, num_classes)

    yhat_probs = model.predict(test_data, verbose=0)
    yhat_classes = model.predict_classes(test_data, verbose=0)
    testy = np.array([int(d) for d in test_label_tmp])
    # f1 = f1_score(testy, np.array(yhat_classes))
    kappa = cohen_kappa_score(testy, yhat_classes)
    # auc = roc_auc_score(testy, yhat_classes)
    matrix = confusion_matrix(testy, yhat_classes)
    acc_score = accuracy_score(testy, yhat_classes)
    print('\n', 'Test accuracy:', acc_score)
    # print("f1 score:", f1)
    print("Cohen's Kappa score: ", kappa)
    # print("auc: ", auc)
    print("confusion matrix: ")
    print(matrix)

    classes = list(labels.values())
    for i, d in enumerate(test_data):
        d_e = (np.expand_dims(d, 0))

        predictions = model.predict(d_e)
        score = tf.nn.softmax(predictions[0])
        print(score)
        print(100 * np.max(score), classes[np.argmax(score)], classes[int(test_label_tmp[i])])
