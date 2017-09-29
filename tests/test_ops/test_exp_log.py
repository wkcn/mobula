import mobula as M
import mobula.operators as O
import numpy as np

def test_exp():
    N, C, H, W = 2,3,4,5
    a = np.random.random((N, C, H, W))
    l = O.Exp(a)
    y = l.eval()
    l.dY = np.random.random(l.Y.shape)
    l.backward()
    exp = np.exp(a)
    assert np.allclose(y, exp)
    assert np.allclose(l.dX, exp * l.dY)

def test_log():
    N, C, H, W = 2,3,4,5
    a = np.random.random((N, C, H, W))
    a[a == 0] = 1.0
    l = O.Log(a)
    y = l.eval()
    l.dY = np.random.random(l.Y.shape)
    l.backward()
    assert np.allclose(y, np.log(a))
    assert np.allclose(l.dX, (1.0 / a) * l.dY)

def test_exp_op():
    N, C, H, W = 2,3,4,5
    X = np.random.random((N, C, H, W))
    assert np.allclose(M.exp(X).eval(), np.exp(X))

def test_log_op():
    N, C, H, W = 2,3,4,5
    X = np.random.random((N, C, H, W))
    X[X == 0] = 1.0
    assert np.allclose(M.log(X).eval(), np.log(X))
