import sys
import os

PATH = os.path.dirname(__file__)
sys.path.append(PATH + "/..")

from mobula.Defines import *
import numpy as np
import random

random.seed(1019)
np.random.seed(1019)


def test_layer_y(layer, X):
    from mobula.layers import Data
    data = Data(X, "data") 
    data.reshape()
    l = layer(data, "testLayer")
    l.reshape()
    data.forward()
    l.forward()
    l.dY = np.ones(l.Y.shape)
    l.backward()
    return l.Y, l.dX
