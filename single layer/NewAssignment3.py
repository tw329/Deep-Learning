import numpy as np
import sys

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

##Initialize weights

w = np.random.rand(hidden_nodes)
print("w =\n", w)

k = np.random.rand(hidden_nodes, cols)
print("W =\n", k)

epochs = 10000
eta = 0.0001
prev_obj = np.inf
i = 0

##Calculate obj

hidden_layer = np.matmul(train, np.transpose(k))
#print("hidden layer =", hidden_layer)
#print("hidden layer shape =", hidden_layer.shape)

sigmoid = lambda x: 1/(1+np.exp(-x))
hidden_layer = np.array([sigmoid(xi) for xi in hidden_layer])
#print("hidden layer =", hidden_layer)
#print("hidden layer shape =", hidden_layer.shape)

output_layer = np.matmul(hidden_layer, np.transpose(w))
#print("output_layer =", output_layer)

obj = np.sum(np.square(output_layer - trainlabels))
print("Start obj =", obj)



##Gradient
stop = 0
count = 0
while(abs(prev_obj - obj) > stop and count < epochs):
    prev_obj = obj
    dellw = (np.dot(hidden_layer[0, :], np.transpose(w)) - trainlabels[0]) * hidden_layer[0, :]
    for j in range(1, rows):
        dellw += (np.dot(hidden_layer[j, :], np.transpose(w)) - trainlabels[j]) * hidden_layer[j, :]

    #print(dellw)
    w = w - eta * dellw

    dell = []
    for i in range(0, hidden_nodes):
        dell.append([])
    for i in range(0, hidden_nodes):
        dell[i] = np.sum(np.dot(hidden_layer[0, :], w) - trainlabels[0]) * w[i] * (hidden_layer[0, i]) * (1 - hidden_layer[0, i]) * train[0]
        for j in range(1, rows):
            dell[i] += np.sum(np.dot(hidden_layer[j, :], w) - trainlabels[j]) * w[i] * (hidden_layer[j, i]) * (1 - hidden_layer[j, i]) * train[j]

    for i in range(0, hidden_nodes):
        k[i] = k[i] - eta * dell[i]

    hidden_layer = np.matmul(train, np.transpose(k))
    hidden_layer = np.array([sigmoid(xi) for xi in hidden_layer])
    output_layer = np.matmul(hidden_layer, np.transpose(w))
    obj = np.sum(np.square(output_layer - trainlabels))

    print("obj =", obj)

    count += 1
print("Final obj =", obj)
test_hidden_layer = hidden_layer
test_output_layer = output_layer
##Predictions
test_hidden_layer = np.matmul(test, np.transpose(k))
test_hidden_layer = np.array([sigmoid(xi) for xi in test_hidden_layer])
test_output_layer = np.matmul(test_hidden_layer, np.transpose(w))
test_obj = np.sum(np.square(test_output_layer - testlabels))
print("test obj =", test_obj)
for i in range(0, len(test_output_layer)):
    if np.abs(test_output_layer[i] - 1) < np.abs(test_output_layer[i] - (-1)) :
        print("1")
    else:
        print("-1")