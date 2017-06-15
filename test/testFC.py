from testlib import *
from mobula.layers import Layer, Data, FC

X = np.zeros((4,2,1,1))
X[0,:,0,0] = [0,0]
X[1,:,0,0] = [0,1]
X[2,:,0,0] = [1,0]
X[3,:,0,0] = [1,1]

Y = np.matrix("8;10;12;14")

data = Data(X)
data.reshape()
data.forward()

fc1 = FC(data, "fc1", dim_out = 1)
fc1.reshape()
fc1.reshape2()

fc1.W = np.matrix("1.,3.")
fc1.b = np.matrix("0.")

lr = 1.0
for i in range(30):
    fc1.forward()
    print "Y^ = ", fc1.Y
    dY = fc1.Y - Y
    fc1.dY = dY
    print "Iter: %d, Cost: %f" % (i, np.sum(np.multiply(dY,dY)))
    fc1.backward()
    fc1.update(lr)

print fc1
print fc1.W, fc1.b

rX = np.matrix("0,0;0,1;1,0;1,1")
dy = (rX * fc1.W.T + fc1.b.T - fc1.Y)
print ("Compute Error: %f" % np.sum(np.multiply(dy,dy)))

#fc2 = FC(fc1, "fc2")
#fc3 = FC(fc2, "fc3")
