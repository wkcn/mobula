from test_layers import *
import mobula.layers as L
from mobula import Net

X = np.arange(60).reshape((5,3,2,2))

l = L.Slice(L.Data(X), "slice", axis = 0, slice_points = [1, 2])

Y1, Y2, Y3 = l()

l.reshape()
l.forward()

a1 = Y1.Y.size
a2 = Y2.Y.size
a3 = Y3.Y.size

Y1.dY = np.arange(0, a1).reshape(Y1.Y.shape)
Y2.dY = np.arange(a1, a1 + a2).reshape(Y2.Y.shape)
Y3.dY = np.arange(a1 + a2, a1 + a2 + a3).reshape(Y3.Y.shape)

l.backward()

print ("X: ", X.shape, X.ravel())
print ("Y1: ", Y1.Y.shape, Y1.Y.ravel())
print ("Y2: ", Y2.Y.shape, Y2.Y.ravel())
print ("Y3: ", Y3.Y.shape, Y3.Y.ravel())

print ("dX: ", l.dX.shape, l.dX.ravel())
