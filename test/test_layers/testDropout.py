from test_layers import *
import mobula
import mobula.layers as L
import time

t = time.clock()
X = np.zeros((2,2,6,6))
X = np.arange(X.size).reshape(X.shape)

data = L.Data(X, "data")
dp = L.Dropout(data, "drop", ratio = 0.3)
dp.net = FakeNet

dp.X = X
dp.reshape()
dp.forward()
print (dp.Y)
print (time.clock() - t)
