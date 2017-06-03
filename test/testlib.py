import sys
sys.path.append("..")
import numpy as np


def test_layer_y(layer, X):
    from mobula.layers import Data
    data = Data(X, "data") 
    l = layer(data, "relu")
    l.forward()
    l.dY = np.ones(l.Y.shape)
    l.backward()
    return l.Y, l.dX
