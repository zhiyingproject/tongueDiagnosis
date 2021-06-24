import tables
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import sys
import keras


hdf5_path = "../datasets/hdf5"
subtract_mean = True
nb_class = 4
epoch = 32

hdf5_file = tables.open_file(hdf5_path, mode='r')

if subtract_mean:
    mm = hdf5_file.root.train_mean[0]
    mm = mm[np.newaxis, ...]

train_data = np.array(hdf5_file.root.train_img)
train_label = np.array(hdf5_file.root.train_labels)

test_data = np.array(hdf5_file.root.test_img)
test_label = np.array(hdf5_file.root.test_labels)



val_data = np.array(hdf5_file.root.val_img)
val_label = np.array(hdf5_file.root.val_labels)

num_classes = len(np.unique(train_label))
train_label = np_utils.to_categorical(train_label, num_classes)
test_label = np_utils.to_categorical(test_label, num_classes)
val_label = np_utils.to_categorical(val_label, num_classes)

# building the model

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', input_shape=(128, 98, 3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='tanh'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='tanh'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(4, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics='accuracy')


checkpointer = ModelCheckpoint(filepath='shape.model.weights.best.hdf5', verbose=1, save_best_only=True)
hist = model.fit(train_data, train_label, batch_size=None, epochs=epoch,
                 validation_data=(val_data, val_label),
                 callbacks=[checkpointer], verbose=1, shuffle=True)

model.load_weights('shape.model.weights.best.hdf5')
class_names = ['正常','胖大舌','瘦薄舌','短缩舌']
class_names = ['淡红舌','淡白舌','淡舌','红绛舌']

for d in test_data:
    d_e = (np.expand_dims(d, 0))

    predictions = model.predict(d_e)
    score = tf.nn.softmax(predictions[0])
    print(100*np.max(score), class_names[np.argmax(score)])
'''for d in test_data:
    print(d.shape)
    prediction = model.predict(d)
    d = tf.expand_dims(d, 0)
    score = tf.nn.softmax(prediction[0])
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
    )
'''

score = model.evaluate(test_data, test_label, verbose=0)


print('\n', 'Test accuracy:', score[1])