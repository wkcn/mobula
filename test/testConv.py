from testlib import *
from mobula.layers import Sigmoid, ReLU, Tanh, Conv
from mobula.layers import Data

X = np.zeros((2,2,5,5))
X[0,0,:,:] = np.arange(25).reshape((5,5))
X[0,1,:,:] = np.arange(25, 50).reshape((5,5))
X[1,0,:,:] = np.arange(125, 150).reshape((5,5))

data = Data(X, "data") 
conv = Conv(data, "Conv", pad = 0, kernel_h = 2, kernel_w = 3, dim_out = 3)
conv.reshape()
conv.forward()
print "X", conv.X.shape
print conv.X_col.shape
conv.dY = conv.Y
print conv.W.T.shape, conv.dY.shape
conv.backward()
#print X
print "==="
#print conv.X_col
print conv.X_col.shape
print conv.dX
'''
print X
print conv.X_col.shape
print conv.W.shape
'''
#print conv.Y.shape
