from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import np_utils, generic_utils, to_categorical
from keras.models import load_model
from keras import backend as k
import random
import matplotlib.pyplot as plt
import os
import numpy as np
import sys

test_set_input = sys.argv[1]
target_model = sys.argv[2]
black_box_model = sys.argv[3]

M = load_model(target_model)
B = load_model(black_box_model)

def gradient(gradientmodel, input_data):

    gradients = k.gradients(gradientmodel.output, gradientmodel.input)
    f = k.function([gradientmodel.input], gradients)
    x = input_data
    
    return f([x])

def generated_adversaries(Dataset, epsilon, model):
    X = np.array(Dataset)
    X = X + random.choice((-1, 1)) * (epsilon) * np.sign(gradient(model, Dataset)[0])
    return X

rows = 32
cols = 32
classes = 2

x_test = np.load(test_set_input+'/data.npy')
x_test = x_test.reshape(x_test.shape[0], rows, cols, 3)
y_test = np.load(test_set_input+'/label.npy')
y_cat = to_categorical(y_test)

NewData = generated_adversaries(x_test, 0.0625, B)
result = M.evaluate(NewData, y_cat)
print("acc :", result[1])
