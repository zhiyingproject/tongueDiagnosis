import tables
import numpy as np
from keras.utils import np_utils
import tensorflow as tf
import sys
sys.path.append('./myClass/')
import preprocessing
import create_model
import os
from tensorflow import keras


categories = ['spike', 'color', 'shape', 'crack', 'coatingColor', 'coatingAmount']
preprocessing_flag = True
original_path = "../originalData/apparatus"
config_path = "../config"
subtract_mean = True
epoch = 16
img_height = 166
img_width = 128
input_shape = (img_height, img_width, 3)
model_name = "iteration1"


for category in categories:
    print("=================================")
    print(f"Training the model for {category}")
    print("=================================")

    hdf5_path = f"../datasets/hdf5{category}"

    if preprocessing_flag:

        data_shape = (0, img_height, img_width, 3)
        re_size = (img_width, img_height)
        preprocessing_class = preprocessing.preprocessing(category, config_path, original_path, hdf5_path, img_width, img_height)
        preprocessing_class.hdf5_generator()

    hdf5_file = tables.open_file(hdf5_path, mode='r')
    if subtract_mean:
        mm = hdf5_file.root.train_mean[0]
        mm = mm[np.newaxis, ...]

    train_data = np.array(hdf5_file.root.train_img)
    train_label = np.array(hdf5_file.root.train_labels)
    val_data = np.array(hdf5_file.root.val_img)
    val_label = np.array(hdf5_file.root.val_labels)

    num_classes = len(np.unique(train_label))

    train_label = np_utils.to_categorical(train_label, num_classes)
    val_label = np_utils.to_categorical(val_label, num_classes)

    # building the model
    checkpoint_path = f'../models/{category}/{model_name}/cp-{epoch}.ckpt'
    model_path = f'../models/{category}/{model_name}'
    checkpoint_dir = os.path.dirname(checkpoint_path)
    batch_size = 32
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
                    filepath=checkpoint_path,
                    verbose=1,
                    save_weights_only=True
                    )
    model = create_model.iteration1(input_shape, num_classes)
    hist = model.fit(train_data, train_label,
                     batch_size=batch_size, epochs=epoch,
                     validation_data=(val_data, val_label),
                     callbacks=[cp_callback], verbose=1)
    model.save(model_path)
    # model.save_weights(checkpoint_path.format(epoch=0))
    hdf5_file.close()