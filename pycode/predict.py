from keras.models import Sequential

model = Sequential()
model.load_weights('model.weights.best.hdf5')