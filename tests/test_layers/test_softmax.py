import mobula.layers as L
import numpy as np

def test_softmax():
    N, C, H, W = 2,3,4,5
    a = np.random.random((N, C, H, W)) - 0.5
    for axis in range(4):
        l = L.Softmax(a, axis = axis)
        y = l.eval()
        exp = np.exp(a)
        su = np.sum(exp, axis = axis)
        axes = [slice(None)] * 4
        axes[axis] = np.newaxis
        pu = [1] * 4
        pu[axis] = a.shape[axis]
        s = np.tile(su[axes], pu)
        assert np.allclose(y, exp / s)
        assert np.allclose(np.sum(y, axis), np.ones(su.shape))
        # backward
        l.dY = np.random.random(l.Y.shape)
        l.backward()
        dX = np.multiply(exp / s - np.square(exp / s), l.dY)
        assert np.allclose(l.dX, dX)
