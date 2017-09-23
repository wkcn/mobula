import mobula as M
import mobula.layers as L
import numpy as np
import operator

def test_compare():
    N, C, H, W = 2,3,4,5
    a = np.random.random((N, C, H, W))
    b = np.random.random((N, C, H, W))
    a_copy = a.copy()
    [la, lb, lac] = L.Data([a,b,a_copy]) 

    ops = [operator.eq, operator.ne, operator.ge, operator.gt, operator.le, operator.lt]
    for op in ops:
        l = op(la, lb)
        l2 = op(la, lac)
        assert np.allclose(l.eval(), op(a,b))
        assert np.allclose(l2.eval(), op(a,a_copy))
        l.backward()
        l2.backward()
        assert np.allclose(l.dX[0], np.zeros(l.X[0].shape)) 
        assert np.allclose(l.dX[1], np.zeros(l.X[1].shape)) 
        assert np.allclose(l2.dX[0], np.zeros(l2.X[0].shape)) 
        assert np.allclose(l2.dX[1], np.zeros(l2.X[1].shape)) 

def test_compare_symbols():
    N, C, H, W = 2,3,4,5
    a = np.random.random((N, C, H, W))
    b = np.random.random((N, C, H, W))
    [la, lb] = L.Data([a,b])
    eq = (la == lb)
    ne = (la != lb)
    ge = (la >= lb)
    gt = (la > lb)
    le = (la <= lb)
    lt = (la < lb)
    
    lops = [eq,ne,ge,gt,le,lt]
    ops = [operator.eq, operator.ne, operator.ge, operator.gt, operator.le, operator.lt]
    for l, op in zip(lops, ops):
        assert l.op == op

def test_compare_symbols_L():
    N, C, H, W = 2,3,4,5
    a = np.random.random((N, C, H, W))
    b = np.random.random((N, C, H, W))
    [la, lb] = L.Data([a,b])
    eq = (la == b)
    ne = (la != b)
    ge = (la >= b)
    gt = (la > b)
    le = (la <= b)
    lt = (la < b)
    
    lops = [eq,ne,ge,gt,le,lt]
    ops = [operator.eq, operator.ne, operator.ge, operator.gt, operator.le, operator.lt]
    for l, op in zip(lops, ops):
        assert l.op == op
