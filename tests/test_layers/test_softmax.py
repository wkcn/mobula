import mobula.layers as L
from mobula.testing import gradcheck
from mobula.layers.utils.Defines import *
import numpy as np

def test_softmax():
    N, C, H, W = 2,3,4,5
    a = np.random.random((N, C, H, W)) - 0.5
    for axis in range(4):
        l = L.Softmax(a, axis = axis)
        label = np.random.randint(0, a.shape[axis], size = a.size // a.shape[axis])
        loss_l = L.SoftmaxWithLoss(a, axis = axis, label = label)

        l.reshape()
        loss_l.reshape()

        y = l.eval()
        loss = loss_l.eval()

        exp = np.exp(a)
        su = np.sum(exp, axis = axis)
        axes = [slice(None)] * 4
        axes[axis] = np.newaxis
        pu = [1] * 4
        pu[axis] = a.shape[axis]
        s = np.tile(su[tuple(axes)], pu)

        # softmax forward
        assert np.allclose(y, exp / s)
        assert np.allclose(np.sum(y, axis), np.ones(su.shape))
        # softmax-with-loss forward
        assert np.allclose(loss_l.softmax, l.Y)
        assert np.allclose(loss_l.Y, -np.mean(np.log(get_val_from_arg(y, label, axis))))
        # softmax backward
        l.dY = np.random.random(l.Y.shape)
        l.backward()
        # softmax-with-loss backward
        loss_l.dY = np.random.random(loss_l.Y.shape)
        loss_l.backward()
        z = np.zeros(y.shape)
        z.ravel()[get_idx_from_arg(z, label, axis)] = 1
        tl = y - z 
        assert np.allclose(tl * loss_l.dY, loss_l.dX)

def test_softmax_grad():
    N, C, H, W = 2, 3, 4, 5
    a = np.random.random((N, C, H, W)) - 0.5
    for axis in range(4):
        l = L.Softmax(a, axis = axis)
        gradcheck(l, a)
