from testlib import *
import mobula.layers as L
import matplotlib.pyplot as plt

X = np.matrix(np.arange(-10,10,0.1)).T
Y, dX = test_layer_y(L.Softmax, X)
print (Y)

plt.subplot(121)
plt.plot(X, Y, 'b')
plt.subplot(122)
plt.plot(X, dX, 'r')
plt.show()
