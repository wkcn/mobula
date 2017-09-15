from mobula.layers.utils.Defines import *
import mobula.operators as O
import numpy as np

def test_reduce_mean():
    N, C, H, W = 2,3,4,5
    a = np.random.random((N, C, H, W))
    l = O.ReduceMean(a, axis = 2)
    y = l.eval()
    l.dY = np.random.random(l.Y.shape)
    l.backward()
    assert np.allclose(y, np.mean(a, 2))
    dY = np.tile(l.dY[:,:,np.newaxis,:], [1,1,4,1]) / 4
    assert np.allclose(l.dX, dY)

def test_reduce_mean2():
    N, C, H, W = 2,3,4,5
    a = np.random.random((N, C, H, W))
    l = O.ReduceMean(a, axis = [1,2])
    y = l.eval()
    l.dY = np.random.random(l.Y.shape)
    l.backward()
    assert np.allclose(y, np.mean(a, (1,2)))
    dY = np.tile(l.dY[:,np.newaxis,np.newaxis,:], [1,3,4,1]) / 12
    assert np.allclose(l.dX, dY)

def test_reduce_max():
    N, C, H, W = 2,3,4,5
    a = np.random.random((N, C, H, W))
    l = O.ReduceMax(a, axis = 2)
    y = l.eval()
    l.dY = np.random.random(l.Y.shape)
    l.backward()
    assert np.allclose(y, np.max(a, 2))
    assert np.allclose(get_val_from_idx(l.dX, l.idx), l.dY.ravel())
    em = l.dX.copy()
    em.ravel()[l.idx] = 0
    assert np.allclose(em, np.zeros(em.shape))


def test_reduce_min():
    N, C, H, W = 2,3,4,5
    a = np.random.random((N, C, H, W))
    l = O.ReduceMin(a, axis = 2)
    y = l.eval()
    l.dY = np.random.random(l.Y.shape)
    l.backward()
    assert np.allclose(y, np.min(a, 2))
    assert np.allclose(get_val_from_idx(l.dX, l.idx), l.dY.ravel())
    em = l.dX.copy()
    em.ravel()[l.idx] = 0
    assert np.allclose(em, np.zeros(em.shape))
