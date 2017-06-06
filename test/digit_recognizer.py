from testlib import *
import csv
from mobula import Net
from mobula.layers import FC, Conv
from mobula.layers import Sigmoid, ReLU, Tanh, CrossEntropy
from mobula.layers import Data
import matplotlib.pyplot as plt

reader = csv.reader(open("./train.csv"))
first = True
X = []
Y = []
k = 0
labels = []
for row in reader:
    if first:
        first = False
        continue
    label = int(row[0])
    x = np.matrix([int(w) for w in row[1:]]).reshape((28,28))
    k += 1
    X.append(x)
    q = np.zeros(10)
    q[label] = 1.0
    Y.append(q)
    labels.append(label)
    '''
    plt.imshow(x, "gray")
    plt.show()
    print x
    '''
    #if k > 1000:
    #    break
print ("Read OK", k)

'''
X = np.zeros((4,1,28,28))

Y = (np.matrix("0;1;1;1"))
'''
X = np.array(X) / 255.0
X.shape = (k,1,28,28)
Y = np.matrix(Y).reshape((k,10)).astype(np.double)
labels = np.array(labels)


data = Data(X, "Data", batch_size = 100, labels = Y)

conv1 = Conv(data, "Conv1", dim_out=20, kernel = 5)

conv2 = Conv(conv1, "Conv2", dim_out=50, kernel = 5)

fc1 = FC(conv2, "fc1", dim_out = 500)
sig1 = Sigmoid(fc1, "relu1")
fc2 = FC(sig1, "fc2", dim_out = 10)
sig2 = Sigmoid(fc2, "sig2")

loss = CrossEntropy(sig2, "Loss", label = data.labels) 
net = Net()
net.setLoss(loss)

net.reshape()
net.reshape2()

net.lr = 0.1
for i in range(10000):
    net.forward()
    pre = np.argmax(sig2.Y,1)
    pre.resize(pre.size)
    right = np.argmax(data.labels, 1).reshape(pre.size)
    bs = (pre == right) 
    b = np.sum(bs)
    print "Accuracy: %f" % (b * 1.0 / len(pre))
    net.backward()
    print "Iter: %d, Cost: %f" % (i, loss.Y)
