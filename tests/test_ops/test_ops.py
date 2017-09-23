import mobula as M
import mobula.layers as L
import numpy as np
import operator

def test_ops():
    N, C, H, W = 2,3,4,5
    a = np.random.random((N, C, H, W))
    b = np.random.random((N, C, H, W))
    assert np.allclose(M.add(a, b).eval(), a + b)
    assert np.allclose(M.subtract(a, b).eval(), a - b)
    assert np.allclose(M.subtract(a, b).eval(), a - b)
    assert np.allclose(M.multiply(a, b).eval(), a * b)
    assert np.allclose(M.reduce_mean(a, axis = 2).eval(), np.mean(a, 2))
    assert np.allclose(M.reduce_max(a, axis = 2).eval(), np.max(a, 2))
    assert np.allclose(M.reduce_min(a, axis = 2).eval(), np.min(a, 2))
    assert np.allclose(M.positive(a).eval(), a)
    assert np.allclose(M.negative(a).eval(), -a)

    R, N, K = 3,4,5
    a = np.random.random((R, N))
    b = np.random.random((N, K))
    assert np.allclose(M.matmul(a, b).eval(), np.dot(a, b))
    assert np.allclose(M.dot(a, b).eval(), np.dot(a, b))

def test_symbols():
    N, C, H, W = 2,3,4,5
    a = np.random.random((N, C, H, W))
    b = np.random.random((N, C, H, W))
    [la, lb] = L.Data([a,b])
    eq = M.equal(la, lb)
    ne = M.not_equal(la, lb)
    ge = M.greater_equal(la, lb)
    gt = M.greater(la, lb)
    le = M.less_equal(la, lb)
    lt = M.less(la, lb)
 
    lops = [eq,ne,ge,gt,le,lt]
    ops = [operator.eq, operator.ne, operator.ge, operator.gt, operator.le, operator.lt]
    for l, op in zip(lops, ops):
        assert l.op == op

