import mobula.layers as L
from mobula.Defines import *
import numpy as np

class Net:
    phase = TRAIN

def test_dropout():
    x = np.random.random((2,4,6,8))
    p = 0.6
    scale = 1.0 / (1.0 - p)
    l = L.Dropout(x, ratio = p)
    # Train
    l.net = Net()
    l.reshape()
    l.forward()
    l.dY = np.random.random(l.Y.shape)
    l.backward()

    b = l.mask
    assert np.allclose(l.Y[b], l.X[b] * scale)
    assert (l.Y[~b] == 0).all()
    assert np.allclose(l.dX[b], l.dY[b] * scale)
    assert (l.dX[~b] == 0).all()

    # Test
    Net.phase = TEST
    l.reshape()
    l.forward()
    l.dY = np.random.random(l.Y.shape)
    l.backward()

    assert np.allclose(l.Y, l.X)
    assert np.allclose(l.dY, l.dX)
