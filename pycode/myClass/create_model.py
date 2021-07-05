from tensorflow.keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import tensorflow_addons as tfa


def tf_models(model_name, input_shape, num_classes):
    if model_name == "iteration1":
        # building the model
        model = Sequential()
        model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='tanh'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='tanh'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.3))
        model.add(Flatten())
        model.add(Dense(500, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam',
                    metrics=[tfa.metrics.CohenKappa(num_classes=num_classes, sparse_labels=False)])

    elif model_name == "iteration2":
        # building the model
        model = Sequential()
        model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        # model.add(Dropout(0.4))
        model.add(Dense(num_classes, activation='softmax'))
        # model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='adam',
                      metrics=[tfa.metrics.CohenKappa(num_classes=num_classes, sparse_labels=False)])

    return model

