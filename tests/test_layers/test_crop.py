import mobula.layers as L
import numpy as np

def test_crop():
    X1 = np.arange(3 * 4 * 5 * 5).reshape((3,4,5,5))
    X2 = np.zeros((3,4,3,3))
    [x1, x2] = L.Data([X1, X2], "data")
    l = L.Crop([x1, x2], "crop", offset = 1, axis = 2)
    l.reshape()
    l.forward()
    target = X1[:,:,1:4,1:4]
    print (target, l.Y)
    assert np.allclose(target, l.Y)
    l.dY = np.arange(l.Y.size).reshape(l.Y.shape)
    l.backward()
    tmp = np.zeros(X1.shape)
    tmp[:,:,1:4,1:4] = l.dY
    assert np.allclose(tmp, l.dX)

    l2 = L.Crop([x1, x2], "crop", offset = [1,1], axis = 2)
    assert (l.offset == l2.offset).all()
