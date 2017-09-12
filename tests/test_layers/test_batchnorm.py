import mobula.layers as L
import numpy as np

def test_batchnorm_mean_var():
    X = np.random.random((12, 10)) * 100
    var = np.var(X, 0)
    mean = np.mean(X, 0)
    data = L.Data(X, "data", batch_size = 4)
    bn = L.BatchNorm(data, "BN")

    data.reshape()
    bn.reshape()
    bn.dY = np.ones(bn.Y.shape)
    rec_mean = []
    rec_var = []
    for i in range(3 * 3):
        data.forward()
        rec_mean.append(np.mean(data.Y, 0))
        rec_var.append(np.var(data.Y, 0))
        bn.forward()
        bn.backward()

    print ("mean", mean, bn.gmean, np.mean(rec_mean, 0))
    assert np.allclose(mean, bn.gmean)
    tvar = np.mean(rec_var, 0)
    print ("var", var, bn.gvar, tvar)
    assert np.allclose(tvar, bn.gvar)

def test_grads():
    X = np.arange(10).reshape((10, 1)).astype(np.float)
    bn = L.BatchNorm(None, "bn")
    bn.X = X
    bn.reshape()
    bn.dY = np.ones(bn.Y.shape)
    bn.W = np.random.random(bn.W.shape) * 10
    bn.b = np.random.random(bn.b.shape) * 10
    bn.forward()
    bn.backward()

    var = np.var(X, 0)
    mean = np.mean(X, 0)

    assert np.allclose(bn.cmean, mean)
    assert np.allclose(bn.cvar, var)

    dnx = bn.dY * bn.W
    dvar = np.sum(dnx * (X - mean), 0) * -0.5 * np.power((var + bn.eps), -1.5)
    dmean = np.sum(dnx * (-1) / np.sqrt(var + bn.eps)) + dvar * np.mean(-2 * (X - mean), 0) 

    dX = dnx * 1.0 / np.sqrt(var + bn.eps) + dvar * 2 * (X - mean) / X.shape[0] + dmean / X.shape[0]
    nx = (X - mean) / np.sqrt(var + bn.eps)
    dW = np.sum(bn.dY * nx, 0)
    db = 10

    #print ("DDDD", dnx, dvar, dmean)
    print ("dX", bn.dX, dX)
    assert np.allclose(bn.dX, dX)
    print ("dW", bn.dW, dW)
    assert np.allclose(bn.dW, dW)
    print ("db", bn.db, db)
    assert np.allclose(bn.db, db)
