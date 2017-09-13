import mobula.layers as L
import numpy as np

def test_multiply():
    N, C, H, W = 2,3,4,5
    a = np.random.random((N, C, H, W))
    b = np.random.random((N, C, H, W))
    [la, lb] = L.Data([a,b])
    w = la * lb
    w.forward()
    w.dY = np.random.random(w.Y.shape)
    w.backward()
    assert np.allclose(np.multiply(a, b), w.Y)
    assert np.allclose(w.dX[0], np.multiply(w.dY, w.X[1]))
    assert np.allclose(w.dX[1], np.multiply(w.dY, w.X[0]))
