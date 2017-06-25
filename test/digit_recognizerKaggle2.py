from testlib import *
import csv
from mobula import Net
import mobula.layers as L
import matplotlib.pyplot as plt
import scipy.io as sio

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
    #if k > 128:
    #    break
    #plt.imshow(x, "gray")
    #plt.show()
    #print x
print ("Read OK", k)
X = np.array(X) / 255.0
X.shape = (k,1,28,28)
Y = np.matrix(Y).reshape((k,10)).astype(np.double)
labels = np.array(labels).reshape((len(labels), 1))


data = L.Data(X, "Data", batch_size = 100, label = labels)

conv1 = L.Conv(data, "Conv1", dim_out = 20, kernel = 5)
pool1 = L.Pool(conv1, "pool1", pool = L.Pool.MAX, kernel = 2, stride = 2)
conv2 = L.Conv(pool1, "Conv2", dim_out = 50, kernel = 5)
pool2 = L.Pool(conv2, "pool2", pool = L.Pool.MAX, kernel = 2, stride = 2)
fc3   = L.FC(pool2, "fc3", dim_out = 500)
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

        vs = []
        for u in range(30):
            net.forward()
            pre = np.argmax(pred.Y,1)
            pre.resize(pre.size)
            ra = data.label.ravel()
            if u % 10 == 0: 
                print ((pre, ra))
            bs = (pre == ra) 
            b = np.mean(bs)
            vs.append(b)
        acc = np.mean(vs)
        print ("Accuracy: %f" % (acc))

print ("Save Over OK")
