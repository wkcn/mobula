import mobula.layers as L
import numpy as np

def test_sign_pos():
    N, C, H, W = 2,3,4,5
    a = np.random.random((N, C, H, W))
    b = np.random.random((N, C, H, W))
    [la, lb] = L.Data([a,b])
    w = (+la)
    w.reshape()
    assert w.Y.shape == a.shape
    assert (+la) is la

def test_sign_neg():
    N, C, H, W = 2,3,4,5
    a = np.random.random((N, C, H, W))
    b = np.random.random((N, C, H, W))
    [la, lb] = L.Data([a,b])
    w = -la
    w.reshape()
    assert w.Y.shape == a.shape
    w.forward()
    w.dY = np.random.random(w.Y.shape)
    w.backward()
    assert np.allclose(w.Y, -(la.Y))
    assert np.allclose(w.dX, -w.dY)
