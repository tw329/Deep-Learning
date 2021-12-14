from keras.models import Sequential
from keras import regularizers
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import np_utils, generic_utils, to_categorical
from keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.models import load_model
import sys
import numpy as np
import keras

batch_size = 80
classes = 10
epoch = 100

channels = 3
rows = 112
cols = 112

train=sys.argv[1]
test=sys.argv[2]
model_name=sys.argv[3]

X_train = np.load(train)
Y_train = np.load(test)
X_train = X_train.reshape(X_train.shape[0], rows, cols, 3)
Y_train = to_categorical(Y_train, classes)


model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', kernel_initializer='he_normal', activation='relu', input_shape=(rows,cols,channels)))
model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', kernel_initializer='he_normal', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', kernel_initializer='he_normal', activation='relu'))
model.add(BatchNormalization(momentum=0.9))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.5))

model.add(Conv2D(filters=128, kernel_size=(5,5), padding='same', kernel_initializer='he_normal', activation='relu'))
model.add(Conv2D(filters=128, kernel_size=(5,5), padding='same', kernel_initializer='he_normal', activation='relu'))
model.add(BatchNormalization(momentum=0.9))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.5))

model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', kernel_initializer='he_normal', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', kernel_initializer='he_normal', activation='relu'))
model.add(BatchNormalization(momentum=0.9))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(64, kernel_initializer='he_normal', activation='relu'))
model.add(Dense(32, kernel_initializer='he_normal', activation='relu'))
model.add(Dense(10, activation='softmax'))

#opt = keras.optimizers.Adadelta(lr=1.0, rho=0.95)
opt = keras.optimizers.Adam(lr=0.005, beta_1=0.9, beta_2=0.999, amsgrad=True)
#opt = keras.optimizers.SGD(lr=0.01, momentum=0.0, nesterov=False)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

def train():
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=epoch,
              shuffle=True)

train()
model.save(model_name)

