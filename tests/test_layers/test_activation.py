import mobula.layers as L
import numpy as np

def test_sigmoid():
    X = ((np.arange(10000) - 5000) / 1000.0).reshape((-1, 1, 1, 1))
    data = L.Data(X, "data")
    data.reshape()
    l = L.Sigmoid(data)
    l.reshape()
    assert l.Y.shape == X.shape
    l.forward()
    l.dY = np.random.random(l.Y.shape) * 10
    l.backward()

    enx = np.exp(-X)
    assert np.allclose(l.Y.ravel(), (1.0 / (1.0 + enx)).ravel())
    assert np.allclose(l.dX.ravel(), (enx / np.square(1 + enx) * l.dY).ravel())

def test_relu():
    X = ((np.arange(10000) - 5000) / 1000.0).reshape((-1, 1, 1, 1))
    data = L.Data(X, "data")
    data.reshape()
    l = L.ReLU(data)
    l.reshape()
    assert l.Y.shape == X.shape
    l.forward()
    l.dY = np.random.random(l.Y.shape) * 10
    l.backward()
    Y = np.zeros(X.shape)
    b = (X > 0)
    Y[b] = X[b]
    dX = np.zeros(X.shape)
    dX[b] = l.dY[b]
    '''
    d = (l.dX != dX)
    print (l.dX[d], dX[d])
    '''
    assert np.allclose(l.Y.ravel(), Y.ravel())
    assert np.allclose(l.dX.ravel(), dX.ravel())

def test_selu():
    X = ((np.arange(10000) - 5000) / 1000.0).reshape((-1, 1, 1, 1))
    data = L.Data(X, "data")
    data.reshape()
    l = L.SELU(data)
    y = l.eval()
    ty = np.zeros(X.shape) 
    ty[X > 0] = l.scale * X[X>0]
    ty[X<=0] = l.scale * (l.alpha * np.exp(X[X<=0]) - l.alpha)
    assert np.allclose(y, ty)
    l.dY = np.random.random(l.Y.shape)
    l.backward()
    dX = np.zeros(X.shape)
    dX[X > 0] = l.scale
    dX[X <= 0] = l.scale * l.alpha * np.exp(X[X<=0])
    dX *= l.dY
    assert np.allclose(dX, l.dX)

def test_PReLU():
    X = ((np.arange(10000) - 5000) / 1000.0).reshape((-1, 1, 1, 1))
    data = L.Data(X, "data")
    data.reshape()
    l = L.PReLU(data)
    y = l.eval()
    ty = np.zeros(X.shape)
    ty[X>0] = X[X>0]
    ty[X<=0] = l.a * X[X<=0] 
    assert np.allclose(y, ty)
    l.dY = np.random.random(l.Y.shape)
    l.backward()
    dX = np.zeros(X.shape)
    dX[X>0] = 1
    dX[X<=0] = l.a
    dX *= l.dY
    print (dX, l.dX)
    assert np.allclose(dX, l.dX)

def test_tanh():
    X = ((np.arange(10000) - 5000) / 1000.0).reshape((-1, 1, 1, 1))
    data = L.Data(X, "data")
    data.reshape()
    l = L.Tanh(data)
    y = l.eval()
    p = np.exp(X)
    n = np.exp(-X)
    ty = (p - n) / (p + n)
    assert np.allclose(y, ty)
    l.dY = np.random.random(l.Y.shape)
    l.backward()
    dX = 1.0 - np.square(p - n) / np.square(p + n)
    dX *= l.dY
    assert np.allclose(dX, l.dX)
