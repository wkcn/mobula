import sys
sys.path.append("..")
import numpy as np
import random

random.seed(1019)
np.random.seed(1019)


def test_layer_y(layer, X):
    from mobula.layers import Data
    data = Data(X, "data") 
    data.reshape()
    data.forward()
    l = layer(data, "testLayer")
    l.forward()
    l.dY = np.ones(l.Y.shape)
    l.backward()
    return l.Y, l.dX
