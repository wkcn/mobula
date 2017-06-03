from testlib import *
from mobula.layers import Layer, Data, FC, Sigmoid, ReLU, MSE, CrossEntropy
from mobula import Net

X = np.zeros((4,2,1,1))
X[0,:,0,0] = [0,0]
X[1,:,0,0] = [0,1]
X[2,:,0,0] = [1,0]
X[3,:,0,0] = [1,1]

Y = np.matrix("0;1;1;0")

data = Data(X, "Data")

fc1 = FC(data, "fc1", dim_out = 2)
sig1 = Sigmoid(fc1, "sig1")
fc2 = FC(sig1, "fc2", dim_out = 1)
sig2 = Sigmoid(fc2, "sig2")
#sig2 = ReLU(fc2, "relu1", dim_out = 1)
#mse = MSE(sig2, "MSE", label = Y)
loss = CrossEntropy(sig2, "Loss", label = Y) 
net = Net()
net.setLoss(loss)

net.reshape()
net.reshape2()

net.lr = 6.0
for i in range(1000):
    net.forward()
    print "Y^ = ", sig2.Y
    net.backward()
    print "Iter: %d, Cost: %f" % (i, loss.Y)

print "fc1:"
print fc1.W, fc1.b
print "fc2:"
print fc2.W, fc2.b
