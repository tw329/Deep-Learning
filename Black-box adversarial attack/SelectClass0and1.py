import matplotlib.pyplot as plt
import os
import numpy as np
import sys

test_data_input = sys.argv[1]
test_label_input = sys.argv[2]
data_output = sys.argv[3]

rows = 32
cols = 32
classes = 10

x_test = np.load(test_data_input)
x_test = x_test.reshape(x_test.shape[0], rows, cols, 3)
y_test = np.load(test_label_input)

target = []
label = []

for i in range(0, len(y_test)):
    if y_test[i] == 0:
        target.append(i)
        label.append(0)
    elif y_test[i] == 1:
        target.append(i)
        label.append(1)


data = np.empty((2000, 32, 32, 3))

for i in range(0, len(target)):
    data[i] = x_test[target[i]]

for i in range(0, len(target)):
    if np.any(data[i] != x_test[target[i]]):
        print("False")

mean = np.mean(data,axis=0)

np.save(data_output+'/data', data)
np.save(data_output+'/label', label)