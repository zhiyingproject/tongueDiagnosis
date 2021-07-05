import tables
import numpy as np
from keras.utils import np_utils
import tensorflow as tf
import sys
sys.path.insert(0, './myClass/')
import preprocessing
import create_model
from collections import Counter


categories = ['spike', 'color', 'shape', 'crack', 'coatingColor', 'coatingAmount']
preprocessing_flag = True
original_path = "../originalData/apparatus"
config_path = "../config"
subtract_mean = True
epoch = 16
img_height = 166
img_width = 128
input_shape = (img_height, img_width, 3)


def get_training_data(category, over_sampling_flag=0):
    hdf5_path = f"../datasets/hdf5{category}"

    if preprocessing_flag:
        print("preprocessing files")
        preprocessing_class = preprocessing.hdf5File(category, config_path, original_path, hdf5_path,
                                                     img_width, img_height)
        preprocessing_class.hdf5_generator()

    hdf5_file = tables.open_file(hdf5_path, mode='r')
    if subtract_mean:
        mm = hdf5_file.root.train_mean[0]
        mm = mm[np.newaxis, ...]

    train_data = np.array(hdf5_file.root.train_img)
    train_label = np.array(hdf5_file.root.train_labels)
    val_data = np.array(hdf5_file.root.val_img)
    val_label = np.array(hdf5_file.root.val_labels)

    hdf5_file.close()
    return train_data, train_label, val_data, val_label


def base_line(model_name):
    '''
    this function trains the model without dealing with unbalanced classes, serving as a baseline
    :param model_name: the name for cnn model
    :return:
    '''
    for category in categories:
        print("=================================")
        print(f"Training the model for {category}")
        print("=================================")

        train_data, train_label, val_data, val_label = get_training_data(category)
        num_classes = len(np.unique(train_label))

        train_label = np_utils.to_categorical(train_label, num_classes)
        val_label = np_utils.to_categorical(val_label, num_classes)

        # building the model
        checkpoint_path = f'../models/{category}/{model_name}/cp-{epoch}.ckpt'
        model_path = f'../models/{category}/{model_name}'
        batch_size = 32
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            verbose=1,
            save_weights_only=True
        )
        model = create_model.tf_models(model_name, input_shape, num_classes)
        hist = model.fit(train_data, train_label,
                         batch_size=batch_size, epochs=epoch,
                         validation_data=(val_data, val_label),
                         callbacks=[cp_callback], verbose=1)
        model.save(model_path)



def over_sampling():

    for category in categories[3:4]:
        print("=================================")
        print(f"Training the model for {category}")
        print("=================================")

        train_data, train_label, val_data, val_label = get_training_data(category)
        train_count = Counter(train_label)
        print(train_count)



if __name__ == '__main__':
    model_name_input = "iteration2"
    base_line(model_name_input)
    # over_sampling()