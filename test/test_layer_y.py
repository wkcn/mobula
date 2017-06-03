from testlib import *
from mobula.layers import Sigmoid, ReLU, Tanh
import matplotlib.pyplot as plt

X = np.matrix(np.arange(-10,10,0.1)).T
Y, dX = test_layer_y(Tanh, X)

plt.subplot(121)
plt.plot(X, Y, 'b')
plt.subplot(122)
plt.plot(X, dX, 'r')
plt.show()
