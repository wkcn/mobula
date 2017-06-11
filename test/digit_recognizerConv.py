from testlib import *
import csv
from mobula import Net
from mobula.layers import FC, Conv
from mobula.layers import Sigmoid, ReLU, Tanh, CrossEntropy
from mobula.layers import Data
import matplotlib.pyplot as plt
import scipy.io as sio

filename = "./ex4data1.mat"
load_data = sio.loadmat(filename)

X = load_data['X']
T = load_data['y']
X.shape = (5000,1,20,20)
T[T == 10] = 0
Y = np.zeros((5000, 10))
for i in range(5000):
    Y[i, T[i]] = 1.0

data = Data(X, "Data", batch_size = 100, labels = Y)

conv1 = Conv(data, "Conv1", dim_out=10, kernel = 5)

conv2 = Conv(conv1, "Conv2", dim_out=20, kernel = 5)

fc1 = FC(conv2, "fc1", dim_out = 50)
sig1 = Sigmoid(fc1, "sig1")
fc2 = FC(sig1, "fc2", dim_out = 10)
sig2 = Sigmoid(fc2, "sig2")

loss = CrossEntropy(sig2, "Loss", label = data.labels) 

net = Net()
net.setLoss(loss)

net.reshape()
net.reshape2()

net.lr = 0.3
for i in range(100000):
    net.forward()
    pre = np.argmax(sig2.Y,1)
    pre.resize(pre.size)
    right = np.argmax(data.labels, 1).reshape(pre.size)
    bs = (pre == right) 
    b = np.sum(bs)
    if (b == len(pre)):
        print (pre, right)
    print "Accuracy: %f" % (b * 1.0 / len(pre))
    net.backward()
    print "Iter: %d, Cost: %f" % (i, loss.Y)
