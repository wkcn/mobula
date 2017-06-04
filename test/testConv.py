from testlib import *
from mobula.layers import Sigmoid, ReLU, Tanh, Conv
from mobula.layers import Data

X = np.zeros((2,2,5,5))
X[0,0,:,:] = np.arange(25).reshape((5,5))
X[0,1,:,:] = np.arange(25, 50).reshape((5,5))
X[1,0,:,:] = np.arange(125, 150).reshape((5,5))

data = Data(X, "data") 
conv = Conv(data, "Conv", pad = 0, kernel = 3, dim_out = 2)
conv.reshape()
conv.forward()
'''
print X
print conv.X_col.shape
print conv.W.shape
'''
print conv.Y.shape
