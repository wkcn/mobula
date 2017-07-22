from test_layers import *
import mobula.layers as L
from mobula import Net

X1 = np.arange(0, 10).reshape((10, 1, 1, 1))
X2 = -X1

data1 = L.Data(X1, 'data1')
data2 = L.Data(X2, 'data2')

merge = L.MergeTest([data1, data2], "merge")

net = Net()
net.setLoss(merge)

net.lr = 1

print ("===before mering===")
print (merge.Y.ravel())
net.forward()
print ("===after mering===")
print (merge.Y.ravel())

merge.dY = np.arange(merge.Y.size).reshape(merge.Y.shape)
net.backward()
print (merge.dY.ravel())
print (data1.dY.ravel())
print (data2.dY.ravel())
