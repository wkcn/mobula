from test_layers import *
import mobula.layers as L
import matplotlib.pyplot as plt

l = L.SmoothL1Loss(None, "loss")

X = np.arange(-10,10,0.1)
X_list = X.tolist()
y_list = []

for x in X_list:
    x_in = np.zeros((1,1,1,1))
    x_in[0,0,0,0] = x
    l.X = x_in
    l.forward()
    y_list.append(l.loss)

Y = np.array(y_list)

l.X = X
l.backward()
dX = l.dX

X.resize(X.size)
Y.resize(Y.size)
dX.resize(dX.size)

plt.subplot(121)
plt.plot(X, Y, 'b')
plt.subplot(122)
plt.plot(X, dX, 'r')
plt.show()
