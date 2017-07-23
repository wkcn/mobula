from test_layers import *
import mobula.layers as L
from mobula import Net

X1 = np.arange(0, 10).reshape((10, 1, 1, 1))
X2 = -X1

data1 = L.Data(X1, 'data1')
data2 = L.Data(X2, 'data2')

merge = L.MergeTest([data1, data2], "merge")

divide = L.SplitTest(merge, "divide")
y1, y2 = divide()

net = Net()
net.setLoss(divide)

net.lr = 1

print ("===before mering===")
print (merge.Y.ravel())
net.forward()
print ("===after mering===")
print (merge.Y.ravel())

#merge.dY = np.arange(merge.Y.size).reshape(merge.Y.shape)
y1.dY = 5 * np.arange(divide.Y[0].size).reshape(divide.Y[0].shape)
divide.dY[1] = -5 * np.arange(divide.Y[1].size).reshape(divide.Y[1].shape)
net.backward()
print ("merge.dY: ", merge.dY.ravel())
print ("data1.dY", data1.dY.ravel())
print ("data2.dY", data2.dY.ravel())

print ("+++++++++++++++++++++++++")
print ("y1.Y", y1.Y.ravel())
print ("y2.Y", y2.Y.ravel())
