import mobula.layers as L
import numpy as np

def test_eval():
    N, C, H, W = 2,3,4,5
    a = np.random.random((N, C, H, W))
    b = np.random.random((N, C, H, W))
    data_a = L.Data()
    data_b = L.Data()
    r_add = data_a + data_b
    r_multiply = data_a * data_b
    r_subtract = data_a - data_b
    datas = {
            data_a: a,
            data_b: b
    }
    assert np.allclose(r_add.eval(datas), a + b)
    assert np.allclose(r_multiply.eval(datas), np.multiply(a,b))
    assert np.allclose(r_subtract.eval(datas), a - b)

    a = np.random.random((N, C, H, W))
    b = np.random.random((N, C, H, W))
    datas = {
            data_a: a,
            data_b: b
    }
    assert np.allclose(r_add.eval(datas), a + b)
    assert np.allclose(r_multiply.eval(datas), np.multiply(a,b))
    assert np.allclose(r_subtract.eval(datas), a - b)

    N,C,H,W = 1,3,2,4
    a = np.random.random((N, C, H, W))
    b = np.random.random((N, C, H, W))
    datas = {
            data_a: a,
            data_b: b
    }
    assert np.allclose(r_add.eval(datas), a + b)
    assert np.allclose(r_multiply.eval(datas), np.multiply(a,b))
    assert np.allclose(r_subtract.eval(datas), a - b)

def test_list_eval():
    N, C, H, W = 2,3,4,5
    a = np.random.random((N, C, H, W))
    b = np.random.random((N, C, H, W))
    l = L.Data([a,b])
    [la, lb] = l
    w = la + lb
    c = np.random.random((N, C, H, W))
    d = np.random.random((N, C, H, W))
    datas = {l:[c, d]}
    np.allclose(c + d, w.eval(datas))

    N,C,H,W = 1,3,4,2
    c = np.random.random((N, C, H, W))
    d = np.random.random((N, C, H, W))
    datas = {l:[c, d]}
    np.allclose(c + d, w.eval(datas))
