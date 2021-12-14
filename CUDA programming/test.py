import random

r1 = random.randint(0, 2)

vector = []
for i in range(0, 1000):
    vector.append(r1)
    r1 = random.randint(0, 2)

with open("vector", "w") as fv:
    for i in vector:
        fv. write("%s " % i)


r2 = random.randint(0, 2)
mat = []
for i in range(0, 1600):
    tmp = []
    for j in range(0, 1000):
        tmp.append(r2)
        r2 = random.randint(0, 2)
    mat.append(tmp)

with open("matrix", "w") as fm:
    for i in range(0, 1600):
        for j in range(0, 1000):
            fm. write("%s" % mat[i][j])
            if j == 999:
                fm.write("\n")
            else:
                fm.write(" ")





import sys
import random

# Read Data

datafile = sys.argv[1]
f = open(datafile)
matrix = []
l = f.readline()

while(l != ''):
    a = l.split()
    l2 = []
    for j in range(0, len(a), 1):
        l2.append(float(a[j]))
    matrix.append(l2)
    l = f.readline()

rows = len(matrix)
cols = len(matrix[0])
f.close()

datafile = sys.argv[2]
f = open(datafile)
vector = []
l = f.readline()

while(l != ''):
    a = l.split()
    l2 = []
    for j in range(0, len(a), 1):
        l2.append(float(a[j]))
    vector.append(l2)
    l = f.readline()

f.close()



# Define inner product
def dot(u, v):
    rows_u = len(u)
    rows_v = len(v)

    if rows_u != rows_v:
        raise ArithmeticError('Error')
    
    sum_dot = 0.0
    for i in range(0, rows_u, 1):
        sum_dot += u[i]*v[i]
    return sum_dot

print(dot(matrix[1], vector[0]))