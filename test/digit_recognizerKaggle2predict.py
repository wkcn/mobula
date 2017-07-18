from testlib import *
import csv
from mobula import Net
import mobula.layers as L
import matplotlib.pyplot as plt
import scipy.io as sio
import os

RESULT_PATH = "./kaggle_mean"

reader = csv.reader(open("./test.csv"))
first = True
X = []
Y = []
k = 0
Xmean = np.load("xmean.npy")
for row in reader:
    if first:
        first = False
        continue
    x = np.matrix([int(w) for w in row]).reshape((28,28))
    k += 1
    X.append(x)
    #if k > 128:
    #    break
    #plt.imshow(x, "gray")
    #plt.show()
    #print x
print ("Read OK", k)
X = np.array(X) 
X.shape = (k,1,28,28)
X = (X - Xmean) / 255.0


data = L.Data(X, "Data", batch_size = 100, label = None)

conv1 = L.Conv(data, "Conv1", dim_out = 20, kernel = 5)
pool1 = L.Pool(conv1, "pool1", pool = L.Pool.MAX, kernel = 2, stride = 2)
conv2 = L.Conv(pool1, "Conv2", dim_out = 50, kernel = 5)
pool2 = L.Pool(conv2, "pool2", pool = L.Pool.MAX, kernel = 2, stride = 2)
fc3   = L.FC(pool2, "fc3", dim_out = 500)
relu3 = L.ReLU(fc3, "relu3")
pred  = L.FC(relu3, "pred", dim_out = 10)

net = Net()
net.setLoss(pred)
iter_num = 10000
net.load(RESULT_PATH + "/kaggle%d.net" % iter_num)

net.lr = 0.1
ok = 0
res = [None] * k
while ok < k:
    net.forward()
    pre = np.argmax(pred.Y,1)
    for d in range(len(data.index)):
        ind = data.index[d]
        if res[ind] == None:
            ok += 1
            print ("%d / %d" % (ok, k))
        res[ind] = pre[d]
fout = open("out_m-%d.csv" % iter_num, "w")
fout.write("ImageId,Label\n")
for i in range(k):
    fout.write("%d,%d\n" % (i + 1, res[i]))
