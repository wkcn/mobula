from testlib import *
import csv
from mobula import Net
import mobula.layers as L
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

data = L.Data(X, "Data", batch_size = 128, label = T)

conv1 = L.Conv(data, "Conv1", dim_out = 30, kernel = 5)
pool1 = L.Pool(conv1, "pool1", pool = L.Pool.MAX, kernel = 2, stride = 2)
#conv2 = L.Conv(pool1, "Conv2", dim_out = 50, kernel = 5)
#pool2 = L.Pool(conv2, "pool2", pool = L.Pool.MAX, kernel = 2, stride = 2)
fc3   = L.FC(pool1, "fc3", dim_out = 50)
relu3 = L.ReLU(fc3, "relu3")
pred  = L.FC(relu3, "pred", dim_out = 10)
loss = L.SoftmaxWithLoss(pred, "loss", label_data = data)

net = Net()
net.setLoss(loss)

net.lr = 0.1
for i in range(100000):
    net.forward()
    net.backward()

    if i % 100 == 0:
        print ("Iter: %d, Cost: %f" % (i, loss.loss))
        net.time()
        old_batch_size = data.batch_size
        data.batch_size = None
        net.reshape()
        net.forward()
        pre = np.argmax(pred.Y,1)
        pre.resize(pre.size)
        bs = (pre == T.ravel()) 
        b = np.sum(bs)
        acc = (b * 1.0 / len(pre))
        print (pre[0:5000:50])
        print ("Accuracy: %f" % (acc))
        data.batch_size = old_batch_size
        net.reshape()

print ("Save Over OK")
