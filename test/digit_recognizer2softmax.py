from testlib import *
import csv
from mobula import Net
from mobula.layers import FC, Conv
from mobula.layers import Sigmoid, ReLU, Tanh, CrossEntropy, SoftmaxWithLoss, Softmax
from mobula.layers import Data
import scipy.io as sio

filename = "./ex4data1.mat"
load_data = sio.loadmat(filename)

X = load_data['X']
T = load_data['y']
X.shape = (5000,1,20,20)
T[T == 10] = 0
'''
Y = np.zeros((5000, 10))
for i in range(5000):
    Y[i, T[i]] = 1.0
'''
Y = T.astype(np.int).reshape((5000, 1, 1, 1))

'''
for i in range(0,5000,100):
    x = X[i,0,:,:]
    plt.imshow(x, "gray")
    plt.show()
'''

data = Data(X, "Data", batch_size = 128, label = Y)

#conv1 = Conv(data, "Conv1", dim_out=10, kernel = 5)

#conv2 = Conv(conv1, "Conv2", dim_out=20, kernel = 5)

'''
fc1 = FC(data, "fc1", dim_out = 100)
sig1 = Sigmoid(fc1, "sig1")
fc2 = FC(sig1, "fc2", dim_out = 10)
sig2 = Sigmoid(fc2, "sig2")

loss = CrossEntropy(sig2, "Loss", label = data.labels) 
'''

fc1 = FC(data, "fc1", dim_out = 15)
sig1 = Sigmoid(fc1, "sig1")
fc2 = FC(sig1, "fc2", dim_out = 10)
sig2 = Sigmoid(fc2, "sig2")
#loss = CrossEntropy(sig2, "Loss", label_data = data)
loss = SoftmaxWithLoss(sig2, "softmax", label_data = data)

net = Net()
net.setLoss(loss)

net.reshape()
net.reshape2()

net.lr = 0.1
for i in range(100000):
    net.forward()
    net.backward()

    if i % 2000 == 0:
        print ("Iter: %d, Cost: %f" % (i, loss.loss))
        net.time()
        old_batch_size = data.batch_size
        data.batch_size = None
        net.reshape()
        net.forward()
        pre = np.argmax(loss.Y, 1)
        pre.resize(pre.size)
        right = T.reshape(5000)#np.argmax(data.label, 1).reshape(pre.size)
        bs = (pre == right) 
        b = np.sum(bs)
        acc = (b * 1.0 / len(pre))
        print (pre[0:5000:50])
        print ("Accuracy: %f" % (acc))
        if b == len(pre):
            np.save("mnist_net9.npy", [fc1.W, fc1.b, fc2.W, fc2.b])
            print ("Save OK")
            import sys
            sys.exit()
        data.batch_size = old_batch_size
        net.reshape()


np.save("mnist_net_over9.npy", [fc1.W, fc1.b, fc2.W, fc2.b])
print ("Save Over OK")
