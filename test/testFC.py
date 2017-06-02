from testlib import *
from mobula.layers import Data, FC

X = np.zeros((4,2,1,1))
X[0,:,0,0] = [0,0]
X[1,:,0,0] = [0,1]
X[2,:,0,0] = [1,0]
X[3,:,0,0] = [1,1]

data = Data(X)

fc1 = FC(data, "fc1", dim_out = 2)
fc1.reshape()
fc1.W = np.matrix("1,2;3,4")
fc1.b = np.matrix("10;20")
fc1.forward()

print fc1

rX = np.matrix("0,0;0,1;1,0;1,1")
print fc1.Y
print fc1.Y.shape
print (rX * fc1.W.T + fc1.b.T == fc1.Y).all()

#fc2 = FC(fc1, "fc2")
#fc3 = FC(fc2, "fc3")
