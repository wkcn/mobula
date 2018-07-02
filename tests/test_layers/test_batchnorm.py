import mobula.layers as L
import numpy as np

def test_batchnorm_mean_var():
    momentum = 0.9
    eps = 1e-5
    use_global_stats = False
    N, C = 12, 10

    X = np.random.random((N, C)) * 100
    var = np.var(X, 0)
    mean = np.mean(X, 0)
    data = L.Data(X, batch_size = 4)
    bn = L.BatchNorm(data, momentum = momentum, eps = eps, use_global_stats = use_global_stats)

    data.reshape()
    bn.reshape()
    bn.dY = np.ones(bn.Y.shape)
    moving_mean = np.zeros((1, C))
    moving_var = np.ones((1, C))
    for i in range(3 * 3):
        data.forward()
        moving_mean = momentum * moving_mean + (1 - momentum) * np.mean(data.Y, 0)
        moving_var = momentum * moving_var + (1 - momentum) * np.var(data.Y, 0)
        bn.forward()
        bn.backward()

    assert np.allclose(moving_mean, bn.moving_mean)
    assert np.allclose(moving_var, bn.moving_var)

    bn.use_global_stats = True

    for i in range(3 * 3):
        data.forward()
        bn.forward()
        bn.backward()

    assert np.allclose(moving_mean, bn.moving_mean)
    assert np.allclose(moving_var, bn.moving_var)

def test_grads():
    momentum = 0.9
    eps = 1e-5
    use_global_stats = False
    X = np.arange(24).reshape((4, 6)).astype(np.float)
    bn = L.BatchNorm(None, momentum = momentum, eps = eps, use_global_stats = use_global_stats)
    bn.X = X
    bn.reshape()
    bn.dY = np.ones(bn.Y.shape)
    bn.W = np.random.random(bn.W.shape) * 10
    bn.b = np.random.random(bn.b.shape) * 10
    bn.forward()
    bn.backward()

    var = np.var(X, 0)
    mean = np.mean(X, 0)

    assert np.allclose(bn.moving_mean, mean * (1 - momentum))
    assert np.allclose(bn.moving_var, var * (1 - momentum) + momentum)

    bn.use_global_stats = True
    bn.moving_var = var = np.random.random(var.shape)
    bn.moving_mean = mean = np.random.random(mean.shape)
    bn.forward()
    bn.backward()

    dnx = bn.dY * bn.W
    dvar = np.sum(dnx * (X - mean), 0) * (-0.5) * np.power((var + bn.eps), -1.5)
    dmean = np.sum(dnx * (-1) / np.sqrt(var + bn.eps), 0) + dvar * np.mean(-2 * (X - mean), 0)

    dX = dnx * 1.0 / np.sqrt(var + bn.eps) + dvar * 2.0 * (X - mean) / X.shape[0] + dmean / X.shape[0]
    nx = (X - mean) / np.sqrt(var + bn.eps)
    dW = np.sum(bn.dY * nx, 0, keepdims = True)
    db = np.sum(bn.dY, 0, keepdims = True)

    print ("dX", bn.dX, dX)
    assert np.allclose(bn.dX, dX)
    print ("dW", bn.dW, dW)
    assert np.allclose(bn.dW, dW)
    print ("db", bn.db, db)
    assert np.allclose(bn.db, db)
