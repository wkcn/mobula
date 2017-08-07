import mobula.layers as L
import numpy as np

def test_reshape():
    X = np.arange(2 * 3 * 4 * 5).reshape((2, 3, 4, 5))
    dY = np.arange(100, 100 + 2 * 3 * 4 * 5).reshape((2, 3, 4, 5))
    l = L.Reshape(None, "reshape", dims = [2, -1, 3, 0])
    l.X = X
    l.reshape()
    l.forward()
    l.dY = dY
    l.backward()

    target = X.reshape((2, 4, 3, 5))
    assert l.Y.shape == target.shape
    assert (l.Y == target).all

    tdX = dY.reshape(X.shape)
    assert l.dX.shape == tdX.shape
    assert (l.dX == tdX).all
