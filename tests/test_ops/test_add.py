import mobula.layers as L
import numpy as np

def test_add():
    N, C, H, W = 2,3,4,5
    a = np.random.random((N, C, H, W))
    b = np.random.random((N, C, H, W))
    c = np.random.random((N, C, H, W))
    l = L.Add([a, b])
    l.reshape()
    l.forward()
    assert np.allclose(a + b, l.Y)

def test_add2():
    N, C, H, W = 2,3,4,5
    a = np.random.random((N, C, H, W))
    b = np.random.random((N, C, H, W))
    c = np.random.random((N, C, H, W))
    data = L.Data([a,b,c])
    [la,lb,lc] = data()
    l = L.Add([la, lb, lc]) 
    l.reshape()
    l.forward()
    assert np.allclose(a + b + c, l.Y)
    l.dY = np.random.random(l.Y.shape)
    l.backward()
    for i in range(3):
        assert np.allclose(l.dX[i], l.dY)

def test_add3():
    N, C, H, W = 2,3,4,5
    a = np.random.random((N, C, H, W))
    b = np.random.random((N, C, H, W))
    c = np.random.random((N, C, H, W))
    data = L.Data([a,b,c])
    [la,lb,lc] = data()
    l = la + lb
    data.reshape()
    l.reshape()
    assert l.shape == a.shape
    data.forward()
    l.forward()
    assert np.allclose(a + b, l.Y)

def test_add4():
    N, C, H, W = 2,3,4,5
    a = np.random.random((N, C, H, W))
    b = np.random.random((N, C, H, W))
    c = np.random.random((N, C, H, W))
    data = L.Data([a,b,c])
    [la,lb,lc] = data()
    l = la + lb + lc
    assert type(l) == L.Add
    assert np.allclose(a + b + c, l.eval())

def test_add_constant():
    N, C, H, W = 2,3,4,5
    a = np.random.random((N, C, H, W))
    data = L.Data(a)
    l = data + 39
    assert type(l) == L.AddConstant
    assert np.allclose(l.eval(), a + 39)
    l.dY = np.random.random(l.Y.shape)
    l.backward()
    assert np.allclose(l.dX, l.dY) 

    l = 10 + data
    assert type(l) == L.AddConstant
    assert np.allclose(l.eval(), a + 10)
    l.dY = np.random.random(l.Y.shape)
    l.backward()
    assert np.allclose(l.dX, l.dY) 
