import numpy as np
import sys

traindir = sys.argv[1]
testdir = sys.argv[2]

labelfile = traindir+'/data.csv'
f = open(labelfile)
trainlabels = {}
l = f.readline()
l = f.readline()
while(l != ''):
    a = l.split(",")
    trainlabels[a[0]] = int(a[1])
    l = f.readline()


labelfile = testdir+'/data.csv'
f = open(labelfile)
testlabels = {}
l = f.readline()
l = f.readline()
while(l != ''):
    a = l.split(",")
    testlabels[a[0]] = int(a[1])
    l = f.readline()


def getList(dict): 
    list = [] 
    for key in dict.keys(): 
        list.append(key) 
          
    return list

names = getList(trainlabels)
labels = []
for i in range(len(names)):
    labels.append(trainlabels[names[i]])

test_names = getList(testlabels)
test_labels = []
for i in range(len(test_names)):
    test_labels.append(testlabels[test_names[i]])

traindata = np.empty((len(trainlabels),3,3))

for i in range(0,len(trainlabels)):
    image_matrix=np.loadtxt(traindir+'/'+names[i])
    traindata[i] = image_matrix

testdata = np.empty((len(testlabels),3,3))

for i in range(0,len(testlabels)):
    image_matrix=np.loadtxt(testdir+'/'+test_names[i])
    testdata[i] = image_matrix

eta = .1
pre_obj = np.inf
i=0

c =np.ones((2,2))
#c =(np.random.rand(2,2) * 2) - 1
print(c) 
sigmoid = lambda x: 1/(1+ np.exp(-x))
obj = 0 

stride = 1
padding = 0

def convolution(input, filter):
    result = np.zeros((2,2))
    for i in range(0, 2, 1):
        for j in range(0, 2, 1):
            result[i][j] = np.sum(input[i:i+2, j:j+2]*filter)
    return result

for i in range(0,len(trainlabels)):
    hidden_layer = convolution(traindata[i],c)
    hidden_layer = 1/(1 + np.exp(-hidden_layer))
    output_layer = np.sum(hidden_layer)/4
    obj += (output_layer - labels[i])**2

print("Initial objective=", obj)

count = 0
stop = 0.01
epoche = 10000
while((pre_obj - obj) > stop ):

    pre_obj = obj

    dellc1 = 0
    dellc2 = 0
    dellc3 = 0
    dellc4 = 0
    f = (output_layer)**0.5

    for i in range(0, len(labels)):

        hidden_layer = convolution(traindata[i],c)
        hidden_layer = 1/(1 + np.exp(-hidden_layer))

        sqrtf = (np.sum(hidden_layer))/4 - labels[i]
        
        c1 = np.zeros((2, 2))
        for a in range(0, 2, 1):
            for b in range(0, 2, 1):
                c1[a][b] = hidden_layer[a][b] * (1 - hidden_layer[a][b]) * traindata[i][a][b]
        
        c2 = np.zeros((2, 2))
        for a in range(0, 2, 1):
            for b in range(0, 2, 1):
                c2[a][b] = hidden_layer[a][b] * (1 - hidden_layer[a][b]) * traindata[i][a][b+1]

        c3 = np.zeros((2, 2))
        for a in range(0, 2, 1):
            for b in range(0, 2, 1):
                c3[a][b] = hidden_layer[a][b] * (1 - hidden_layer[a][b]) * traindata[i][a+1][b]

        c4 = np.zeros((2, 2))
        for a in range(0, 2, 1):
            for b in range(0, 2, 1):
                c4[a][b] = hidden_layer[a][b] * (1 - hidden_layer[a][b]) * traindata[i][a+1][b+1]

        dellc1 += sqrtf * (np.sum(c1))/2
        dellc2 += sqrtf * (np.sum(c2))/2
        dellc3 += sqrtf * (np.sum(c3))/2
        dellc4 += sqrtf * (np.sum(c4))/2

    c[0][0] -= eta*dellc1
    c[0][1] -= eta*dellc2
    c[1][0] -= eta*dellc3
    c[1][1] -= eta*dellc4

    obj = 0 
    for i in range(0,len(labels)):
        hidden_layer = convolution(traindata[i],c)
        hidden_layer = 1/(1 + np.exp(-hidden_layer))
        output_layer = np.sum(hidden_layer)/4
        obj += (output_layer - labels[i])**2

    print("objective=", obj)
    count += 1

print("Final kernel(c)=", c)


print("Predictions:")
for i in range(0,len(test_names)):
    hidden_layer = convolution(testdata[i],c)
    hidden_layer = 1/(1 + np.exp(-hidden_layer))
    output_layer = np.sum(hidden_layer)/4
    print(output_layer)
    if(output_layer > 0.5):
    	print(test_names[i], "1")
    else:
    	print(test_names[i], "-1")
