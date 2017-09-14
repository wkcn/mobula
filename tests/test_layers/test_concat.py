import mobula
import mobula.layers as L
import numpy as np

def test_concat():
    def go_concat(axis):
        a = np.random.random((2,3,4,5))
        b = np.random.random((2,3,4,5))
        c = np.random.random((2,3,4,5))
        l = L.Concat([a,b,c], axis = axis)
        l.reshape()
        y = l.eval()
        l.dY = np.random.random(l.Y.shape)
        l.backward()
        assert np.allclose(y, np.concatenate([a,b,c], axis))
        dd = a.shape[axis]
        sa = slice(None, dd)
        sb = slice(dd, dd * 2)
        sc = slice(dd * 2, dd * 3)
        def G(s):
            w = [slice(None) for _ in range(4)]
            w[axis] = s
            return w
        ssa = G(sa)
        ssb = G(sb)
        ssc = G(sc)
        assert np.allclose(l.dY[ssa], l.dX[0]) 
        assert np.allclose(l.dY[ssb], l.dX[1]) 
        assert np.allclose(l.dY[ssc], l.dX[2]) 

    for axis in range(4):
        go_concat(axis = axis)
