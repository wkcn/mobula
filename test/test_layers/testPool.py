from test_layers import *
import mobula.layers as L
from mobula import Net

X = np.zeros((2,2,6,6))
X[0,0,2,2] = 1.0
X[0,0,1,2] = 10.0
X[0,0,2,3] = 20.0
X[0,1,2,2] = 1.0
X[1,0,2,2] = 2.0

data = L.Data(X, "data")
pool = L.Pool(data, "Pool", kernel = 3, stride = 3, pool = L.Pool.AVG)
pool.X = X
pool.reshape()
pool.forward()
pool.dY = pool.Y + 0
pool.backward()

print ("X:")
print (pool.X)

print ("Y:")
print (pool.Y)

print ("dX:")
print (pool.dX)
