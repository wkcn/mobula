import mobula as M
import mobula.layers as L
import numpy as np
import os

def clear_params(layer):
    rec = []
    for i in range(len(layer.params)):
        rec.append(layer.params[i])
        layer.params[i][...] = None
    return rec

def init_params(layer):
    for i in range(len(layer.params)):
        layer.params[i][...] = np.random.random(layer.params[i].shape) 

def test_net_saver():
    filename = "tmp.net"

    X = np.random.random((4,2,1,1))
    Y = np.random.random((4, 10))
    x, y = L.Data([X, Y])
    x = L.FC(x, dim_out = 10) 
    with M.name_scope("mobula"): 
        x = L.PReLU(x)
    loss = L.MSE(x, label = y)

    net = M.Net()
    net.set_loss(loss)
    net.lr = 0.01

    for i in range(10):
        net.forward()
        net.backward()

    net.save(filename)
    # random init layers
    lst = M.get_layers("/")
    assert len(lst) == 4 # Data, FC, PReLU, MSE 

    k = 0
    rec = []
    for l in lst:
        for i in range(len(l.params)):
            rec.append(l.params[i])
            l.params[i][...] = None
            k += 1
    assert k == 3 # FC.W, FC.b, PReLU.a 

    for l in lst:
        for i in range(len(l.params)):
            assert np.isnan(l.params[i]).all()

    net.load(filename)
    h = 0
    for l in lst:
        for i in range(len(l.params)):
            assert np.allclose(rec[h], l.params[i])
            h += 1
    os.remove(filename)

def test_saver():
    filename = "tmp.net"

    X = np.random.random((4,2,1,1))
    Y = np.random.random((4, 10))
    x, y = L.Data([X, Y])
    fc = L.FC(x, dim_out = 10) 
    with M.name_scope("mobula"): 
        prelu = L.PReLU(fc)
    loss = L.MSE(prelu, label = y)

    net = M.Net()
    net.set_loss(loss)

    init_params(fc)
    init_params(prelu)
    # save mobula
    M.save_scope(filename, "mobula")

    params_f = clear_params(fc)
    params_p = clear_params(prelu)
    for p in fc.params + prelu.params:
        assert np.isnan(p).all()
    M.load_scope(filename)
    for p in fc.params:
        assert np.isnan(p).all()
    for i, p in enumerate(prelu.params):
        assert np.allclose(p, params_p[i])

    init_params(fc)
    init_params(prelu)
    # save all
    M.save_scope(filename)

    params_f = clear_params(fc)
    params_p = clear_params(prelu)

    for p in fc.params + prelu.params:
        assert np.isnan(p).all()
    M.load_scope(filename)
    for i, p in enumerate(fc.params):
        assert np.allclose(p, params_f[i])
    for i, p in enumerate(prelu.params):
        assert np.allclose(p, params_p[i])
    os.remove(filename)
