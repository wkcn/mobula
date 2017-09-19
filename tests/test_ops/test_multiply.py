import mobula.layers as L
import numpy as np

def test_multiply():
    N, C, H, W = 2,3,4,5
    a = np.random.random((N, C, H, W))
    b = np.random.random((N, C, H, W))
    [la, lb] = L.Data([a,b])
    w = la * lb
    w.reshape()
    assert w.Y.shape == a.shape
    w.forward()
    w.dY = np.random.random(w.Y.shape)
    w.backward()
    assert np.allclose(np.multiply(a, b), w.Y)
    assert np.allclose(w.dX[0], np.multiply(w.dY, w.X[1]))
    assert np.allclose(w.dX[1], np.multiply(w.dY, w.X[0]))

def test_mul():
    N, C, H, W = 2,3,4,5
    a = np.random.random((N, C, H, W))
    data = L.Data(a)
    l = data * 3
    assert type(l) == L.MultiplyConstant
    assert np.allclose(a * 3, l.eval())
    l.dY = np.random.random(l.Y.shape)
    l.backward()
    assert np.allclose(l.dX, 3 * l.dY)
    l = 3 * data
    assert type(l) == L.MultiplyConstant
    assert np.allclose(a * 3, l.eval())
    l.dY = np.random.random(l.Y.shape)
    l.backward()
    assert np.allclose(l.dX, 3 * l.dY)

