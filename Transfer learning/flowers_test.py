from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential,Model,load_model
from keras.layers import Dropout, Flatten, Dense,Input
from keras.utils import np_utils, generic_utils, to_categorical
from keras.models import load_model
from keras.utils import np_utils
from keras.utils import Sequence
import os
from PIL import Image
from PIL import ImageFilter
import numpy as np
import pandas as pd
import os
from keras.layers import Input
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing.image import ImageDataGenerator
import sys
import keras
val_dir = sys.argv[1]
model_input = sys.argv[2]

test_gen = ImageDataGenerator(rescale=1/255)

test_data = test_gen.flow_from_directory(val_dir, target_size=(331, 331), batch_size=40, class_mode='categorical')

model = load_model(model_input)

result = model.evaluate_generator(test_data)
print("Accuracy =", result[1] * 100, "%")
print("test error", 1-result[1])