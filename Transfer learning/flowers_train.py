from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential,Model,load_model
from keras.layers import Dropout, Flatten, Dense,Input, GlobalAveragePooling2D, GlobalMaxPooling2D, BatchNormalization
from keras.utils import np_utils, generic_utils, to_categorical
from keras.models import load_model
import os
import sys
import keras
train_dir = sys.argv[1]
model_output = sys.argv[2]

train_gen = ImageDataGenerator(rescale=1/255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

train_data = train_gen.flow_from_directory(train_dir, target_size=(331, 331), batch_size=32, class_mode='categorical')

base_model = applications.Xception(include_top=False, weights='imagenet', input_shape=(331, 331, 3))

for layer in base_model.layers:
    layer.trainable=False


output = base_model.output
output = Dropout(0.5)(output)
output = BatchNormalization(momentum=0.9)(output)
output = GlobalAveragePooling2D()(output)
output = Dense(5, activation='softmax')(output)

model = Model(inputs = base_model.input, outputs = output)
#opt = keras.optimizers.Adam(lr=0.001, decay=1e-7)
opt = keras.optimizers.Adagrad(lr=0.01, decay=1e-7)
model.compile(optimizer = 'adamax', loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.fit_generator(train_data, steps_per_epoch=150, epochs=15)

model.save(model_output)
