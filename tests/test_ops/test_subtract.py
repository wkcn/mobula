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

def test_subtract_op_l():
    N, C, H, W = 2,3,4,5
    a = np.random.random((N, C, H, W))
    data = L.Data(a)
    l = data - 3
    assert type(l) == L.SubtractConstantL
    assert np.allclose(a - 3, l.eval())
    l.dY = np.random.random(l.Y.shape)
    l.backward()
    assert np.allclose(l.dX, l.dY)

def test_subtract_op_r():
    N, C, H, W = 2,3,4,5
    a = np.random.random((N, C, H, W))
    data = L.Data(a)
    l = 3 - data
    print (type(l))
    assert type(l) == L.SubtractConstantR
    assert np.allclose(3 - a, l.eval())
    l.dY = np.random.random(l.Y.shape)
    l.backward()
    assert np.allclose(l.dX, -l.dY)
