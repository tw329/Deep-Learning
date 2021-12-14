import numpy as np
import sys
import random

## Read data

f = open(sys.argv[1])
data = np.loadtxt(f)
train = data[:, 1:]
trainlabels = data [:, 0]

onearray = np.ones((train.shape[0], 1))
train = np.append(train, onearray, axis=1)

print("train =", train)
print("train shape =", train.shape)

f = open(sys.argv[2])
data = np.loadtxt(f)
test = data[:, 1:]
testlabels = data[:, 0]
onearray = np.ones((test.shape[0], 1))
test = np.append(test, onearray, axis=1)

rows = train.shape[0]
cols = train.shape[1]
print("rows =", rows, "\ncols =", cols)

hidden_nodes = int(sys.argv[3])
mini_batch = int(sys.argv[4])

##Initialize weights

#w = random.sample(list(np.arange(np.amin(train), np.amax(train), 0.01)), hidden_nodes)
w = np.random.rand(hidden_nodes)
print("w =\n", w)

k = np.random.rand(hidden_nodes, cols)
print("W =\n", k)

epochs = 1000
eta = 0.01

##Calculate obj


hidden_layer = np.matmul(train, np.transpose(k))
#print("hidden layer =", hidden_layer)
#print("hidden layer shape =", hidden_layer.shape)


sigmoid = lambda x: 1/(1+np.exp(-x))
#sign = lambda x: np.sign(x)
hidden_layer = np.array([sigmoid(xi) for xi in hidden_layer])
#hidden_layer = np.array([sign(xi) for xi in hidden_layer])
#print("hidden layer =", hidden_layer)
#print("hidden layer shape =", hidden_layer.shape)

output_layer = np.matmul(hidden_layer, np.transpose(w))
#print("output_layer =", output_layer)

obj = np.sum(np.square(output_layer - trainlabels))
print("Start obj =", obj)



##Gradient
e = 0
while e < epochs:
    
    dellw = 0
    dell = [0] * hidden_nodes
    for lp in range(0, rows):
        randomlist = random.sample(range(0, rows), mini_batch)
        dellw = (np.dot(hidden_layer[randomlist[0], :], np.transpose(w)) - trainlabels[randomlist[0]]) * hidden_layer[randomlist[0], :]
        for j in range(1, len(randomlist)):
            dellw += (np.dot(hidden_layer[randomlist[j], :], np.transpose(w)) - trainlabels[randomlist[j]]) * hidden_layer[randomlist[j], :]
    
        #print(dellw)
        w = w - eta * dellw

        for i in range(0, hidden_nodes):
            dell[i] = np.sum(np.dot(hidden_layer[randomlist[0], :], w) - trainlabels[randomlist[0]]) * w[i] * (hidden_layer[randomlist[0], i]) * (1 - hidden_layer[randomlist[0], i]) * train[randomlist[0]]
            for j in range(1, len(randomlist)):
                dell[i] += np.sum(np.dot(hidden_layer[randomlist[j], :], w) - trainlabels[randomlist[j]]) * w[i] * (hidden_layer[randomlist[j], i]) * (1 - hidden_layer[randomlist[j], i]) * train[randomlist[j]]

        for i in range(0, hidden_nodes):
            k[i] = k[i] - eta * dell[i]
        if e == 1 and lp == 1:
            print(dell)
    hidden_layer = np.matmul(train, np.transpose(k))
    hidden_layer = np.array([sigmoid(xi) for xi in hidden_layer])
    #hidden_layer = np.array([sign(xi) for xi in hidden_layer])
    output_layer = np.matmul(hidden_layer, np.transpose(w))
    obj = np.sum(np.square(output_layer - trainlabels))
    e = e + 1
    print("obj =", obj)
print("Final obj =", obj)
##Predictions

test_hidden_layer = np.matmul(test, np.transpose(k))
test_hidden_layer = np.array([sigmoid(xi) for xi in test_hidden_layer])
test_output_layer = np.matmul(test_hidden_layer, np.transpose(w))
test_obj = np.sum(np.square(test_output_layer - testlabels))
print("test obj =", test_obj)
print("w =", w)
for i in range(0, len(test_output_layer)):
    if np.abs(test_output_layer[i] - 1) < np.abs(test_output_layer[i] - (-1)) :
        print(i, "1")
    else:
        print(i, "-1")