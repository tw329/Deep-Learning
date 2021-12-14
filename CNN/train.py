from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential,Model,load_model
from keras.layers import Dropout, Flatten, Dense,Input
from keras.utils import np_utils, generic_utils, to_categorical
from keras.models import load_model
import os
import sys
import keras
train_dir = sys.argv[1]
model_output = sys.argv[2]

train_gen = ImageDataGenerator(rescale=1/255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

train_data = train_gen.flow_from_directory(train_dir, target_size=(299, 299), batch_size=32, class_mode='categorical')

base_model = applications.Xception(include_top=False, weights='imagenet', input_shape=(299, 299, 3))

for layer in base_model.layers:
    layer.trainable=False

output = base_model.output
output = Dropout(0.3)(output)
output = Flatten()(output)
output = Dense(256, activation ='relu')(output)
output = Dropout(0.3)(output)
output = Dense(128, activation ='relu')(output)
output = Dropout(0.3)(output)
output = Dense(64, activation ='relu')(output)
output = Dropout(0.3)(output)
output = Dense(10, activation='softmax')(output)

model = Model(inputs=base_model.input, outputs=output)
#opt = keras.optimizers.Adagrad(lr=0.1,decay=0)
opt = keras.optimizers.Adagrad(lr=0.01, decay=1e-6)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(train_data, steps_per_epoch=2000, epochs=4)

model.save(model_output)
