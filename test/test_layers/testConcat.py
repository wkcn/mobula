from test_layers import *
import mobula.layers as L
from mobula import Net

X1 = np.arange(12).reshape((2,3,2,1))
X2 = np.arange(12, 12 + 12).reshape((2,3,2,1))

for i in range(4):
    print ("axis = %d:" % i)
    l = L.Concat([L.Data(X1), L.Data(X2)], "concat", axis = i)

    l.reshape()
    l.forward()

    l.dY = l.Y 

    l.backward()
    print ("X1: ", X1.shape, X1.ravel())
    print ("X2: ", X2.shape, X2.ravel())
    print ("y: ", l.Y.shape, l.Y.ravel())
    for j in range(len(l.X)):
        print ("dX[%d]: " % j, l.dX[j].ravel())
