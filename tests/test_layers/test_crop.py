import mobula.layers as L
import numpy as np

def test_crop():
    X1 = np.arange(4 * 5 * 6 * 6).reshape((4,5,6,6))
    for offset in range(2): 
        for axis in range(4):
            to_shp = list(X1.shape)
            w = [slice(None)] * 4
            for i in range(axis, 4):
                to_shp[i] -= 2
                w[i] = slice(offset, to_shp[i] + offset)

            X2 = np.zeros(to_shp)
            [x1, x2] = L.Data([X1, X2])
            l = L.Crop([x1, x2], offset = offset, axis = axis)
            l.reshape()
            l.forward()
            assert np.allclose(l.Y, X1[w]) 
            l.dY = np.random.random(l.Y.shape)
            l.backward()
            tmp = np.zeros(X1.shape)
            tmp[w] = l.dY
            assert np.allclose(l.dX, tmp)

def test_crop2():
    X1 = np.arange(3 * 4 * 5 * 5).reshape((3,4,5,5))
    X2 = np.zeros((2,2,2,2))
    [x1, x2] = L.Data([X1, X2])
    l = L.Crop([x1, x2], offset = (1,2,1), axis = 1)
    assert np.allclose(l.eval(), X1[:, 1:3, 2:4, 1:3])
    l.dY = np.random.random(l.Y.shape)
    l.backward()
    dX = np.zeros(X1.shape) 
    dX[:,1:3,2:4,1:3] = l.dY
    assert np.allclose(l.dX, dX)
