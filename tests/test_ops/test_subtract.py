import mobula.layers as L
import numpy as np

def test_subtract():
    N, C, H, W = 2,3,4,5
    a = np.random.random((N, C, H, W))
    b = np.random.random((N, C, H, W))
    [la, lb] = L.Data([a,b])
    w = la - lb
    w.reshape()
    assert w.Y.shape == a.shape
    w.forward()
    w.dY = np.random.random(w.Y.shape)
    w.backward()
    assert np.allclose(a-b, w.Y)
    assert np.allclose(w.dX[0], w.dY)
    assert np.allclose(w.dX[1], -w.dY)
