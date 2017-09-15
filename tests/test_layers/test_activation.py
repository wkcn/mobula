import mobula.layers as L
import numpy as np

X = ((np.arange(10000) - 5000) / 1000.0).reshape((-1, 1, 1, 1))
data = L.Data(X, "data")
data.reshape()

def test_sigmoid():
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
