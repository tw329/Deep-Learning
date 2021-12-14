from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.utils import np_utils, generic_utils, to_categorical
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
import sys
import numpy as np
import keras

batch_size = 64
classes = 10
epoch = 10

channels = 3
rows = 112
cols = 112

train=sys.argv[1]
test=sys.argv[2]
mode=sys.argv[3]

X_train = np.load(train)
Y_train = np.load(test)
X_train = X_train.reshape(X_train.shape[0], rows, cols, 3)
Y_train = to_categorical(Y_train, classes)

model = Sequential()

model.add(Conv2D(32, (5, 5), activation='relu', padding='same', input_shape=(rows, cols, channels)))
model.add(Conv2D(32, (5, 5), activation='relu', padding='same', input_shape=(rows, cols, channels)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu', padding = 'same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding = 'same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(classes, activation='softmax'))

opt = keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=1e-6)


model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

def train():
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=epoch,
              shuffle=True)

train()
model.save(mode)

