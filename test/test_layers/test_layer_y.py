from test_layers import *
import mobula.layers as L
import matplotlib.pyplot as plt

X = np.arange(-10,10,0.1)
X.resize((X.size, 1, 1, 1))
Y, dX = test_layer_y(L.SELU, X)

X.resize(X.size)
Y.resize(Y.size)
dX.resize(dX.size)

print (dX)
plt.subplot(121)
plt.plot(X, Y, 'b')
plt.subplot(122)
plt.plot(X, dX, 'r')
plt.show()
