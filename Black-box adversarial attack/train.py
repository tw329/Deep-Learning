from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense,Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import np_utils, generic_utils, to_categorical
from keras import backend as k
import tensorflow as tf
import os
import numpy as np
import random
import sys

test_set_input = sys.argv[1]
target_model = sys.argv[2]
black_box_model = sys.argv[3]

rows = 32
cols = 32
classes = 2

x_test = np.load(test_set_input+'/data.npy')
x_test = x_test.reshape(x_test.shape[0], rows, cols, 3)
y_test = np.load(test_set_input+'/label.npy')
y_cat = to_categorical(y_test)

M = load_model(target_model)
result = M.evaluate(x_test, y_cat)

print("Target model accuracy without adversaries :", result[1])

def predict(targetmodel, dataset_pre):
    Mx = targetmodel.predict(dataset_pre)
    y = []
    for i in range(len(Mx)):
        if np.any(Mx[i] == 1):
            y.append(1)
        else:
            y.append(-1)      
    return y

def create_B():
    B = Sequential()
    B.add(Dense(100, activation='tanh', input_shape=(32, 32, 3)))
    B.add(Dense(100, activation='tanh'))
    B.add(Flatten())
    B.add(Dense(1, activation='tanh'))
    B.compile(loss='binary_crossentropy', optimizer='adam',metrics = ['accuracy'])
    return B

def gradient2(model, data):
    gradients = k.gradients(model.output, model.input)[0]
    sess=tf.compat.v1.InteractiveSession()
    sess.run(tf.compat.v1.initialize_all_variables())
    eval_gradients = sess.run(gradients, feed_dict={model.input:data})
    return eval_gradients

'''
def gradient(attackmodel, input_data):

    gradients = k.gradients(attackmodel.layers[-2].output, attackmodel.input)
    f = k.function([attackmodel.input], gradients)
    x = input_data
    
    return f([x])
'''

def generated_adversaries(InputData, epsilon, model):
    adversary = (random.choice([-1, 1])) * (epsilon) * (gradient2(model, InputData)[0])
    output = np.array(InputData) + adversary
    return output



# Initialize 200 random datapoint for D
r200 = random.sample(range(0, len(x_test)), 200)
D = np.empty((200, 32, 32, 3))
label = np.empty((200, 1))
for i in range(len(r200)):
    D[i] = x_test[r200[i]]
    label[i] = y_test[r200[i]]

#print(to_categorical(label))

test_data = np.empty(((len(x_test) - len(D)), 32, 32, 3))
test_label = np.empty(((len(x_test) - len(D)), 1))
j = 0
for i in range(len(x_test)):
    if i not in r200:
        test_data[j] = x_test[i]
        test_label[j] = y_test[i]
        j += 1

B = create_B()

epoch = 20
i = 0

print(gradient2(B, D))

for i in range(epoch):
    y = predict(M, D)
    B.fit(D, y, verbose=2)
    NewData = generated_adversaries(D, .9, B)
    if len(NewData) >= 10000:
        D = NewData
        label = label
    else:
        conbin_data = np.concatenate((D, NewData), axis=0)
        D = conbin_data
        conbin_label = np.concatenate((label, label), axis=0)
        label = conbin_label
    result = M.evaluate(D, to_categorical(label))
    print("Target model accuracy after epoch", i+1, ":", result[1])


result = M.evaluate(generated_adversaries(test_data, 0.0625, B), to_categorical(test_label))
print("Final acc :", result[1])
B.save(black_box_model)