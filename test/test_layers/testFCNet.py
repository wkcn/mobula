from test_layers import *
from mobula.layers import Layer, Data, FC, MSE
from mobula import Net

X = np.zeros((4,2,1,1))
X[0,:,0,0] = [0,0]
X[1,:,0,0] = [0,1]
X[2,:,0,0] = [1,0]
X[3,:,0,0] = [1,1]

Y = np.matrix("8;10;12;14")

data = Data(X, "Data", label = Y)

fc1 = FC(data, "fc1", dim_out = 1)
mse = MSE(fc1, "MSE", label_data = data)
net = Net()
net.setLoss(mse)

fc1.W = np.matrix("1.,3.")
fc1.b = np.matrix("0.")

net.lr = 0.5
for i in range(30):
    net.forward()
    print ("Y = ", fc1.Y)
    net.backward()
    print ("Iter: %d, Cost: %f" % (i, mse.Y))

print (fc1)
print (fc1.W, fc1.b)

rX = np.matrix("0,0;0,1;1,0;1,1")
dy = (rX * fc1.W.T + fc1.b.T - fc1.Y)
print ("Compute Error: %f" % np.sum(np.multiply(dy,dy)))
