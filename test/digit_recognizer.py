from testlib import *
import csv
from mobula import Net
from mobula.layers import FC, Conv
from mobula.layers import Sigmoid, ReLU, Tanh, CrossEntropy
from mobula.layers import Data


X = np.zeros((4,3,28,28))

Y = (np.matrix("0;1;1;1"))

data = Data(X, "Data")

conv1 = Conv(data, "Conv1", dim_out=20, kernel = 5)

conv2 = Conv(conv1, "Conv2", dim_out=50, kernel = 5)

fc1 = FC(conv2, "fc1", dim_out = 500)
relu1 = ReLU(fc1, "relu1")
fc2 = FC(relu1, "fc2", dim_out = 10)
sig2 = Sigmoid(fc2, "sig2")

loss = CrossEntropy(sig2, "Loss", label = Y) 
net = Net()
net.setLoss(loss)

net.reshape()
net.reshape2()

net.lr = 0.1
for i in range(10000):
    net.forward()
    net.backward()
    print "Iter: %d, Cost: %f" % (i, loss.Y)
