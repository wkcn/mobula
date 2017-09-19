import mobula as M
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

def test_matmul():
    R, N, K = 3,4,5
    a = np.random.random((R, N))
    b = np.random.random((N, K))
    [la, lb] = L.Data([a,b])
    l = L.MatMul([la, lb])
    assert np.allclose(l.eval(), np.dot(a, b))
    l.dY = np.random.random(l.Y.shape)
    l.backward()
    assert np.allclose(l.dX[0], np.dot(l.dY, b.T)) 
    assert np.allclose(l.dX[1], np.dot(a.T, l.dY)) 
    # test constant
    lac = M.dot(la, b)
    lbc = M.dot(a, lb)
    assert np.allclose(lac.eval(), np.dot(a, b))
    assert np.allclose(lbc.eval(), np.dot(a, b))
    lac.dY = np.random.random(lac.Y.shape)
    lbc.dY = np.random.random(lbc.Y.shape)
    lac.backward()
    lbc.backward()
    assert np.allclose(lac.dX, np.dot(lac.dY, b.T))
    assert np.allclose(lbc.dX, np.dot(a.T, lbc.dY))
