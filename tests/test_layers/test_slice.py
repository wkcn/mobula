import mobula
import mobula.layers as L
import numpy as np

def test_slice():
    X = np.arange(60).reshape((5,3,2,2))

    l = L.Slice(L.Data(X), "slice", axis = 0, slice_points = [1, 2])

    Y1, Y2, Y3 = l()

    l.reshape()
    l.forward()

    a1 = Y1.Y.size
    a2 = Y2.Y.size
    a3 = Y3.Y.size

    Y1.dY = np.arange(0, a1).reshape(Y1.Y.shape)
    Y2.dY = np.arange(a1, a1 + a2).reshape(Y2.Y.shape)
    Y3.dY = np.arange(a1 + a2, a1 + a2 + a3).reshape(Y3.Y.shape)

    l.backward()

    print ("X: ", X.shape, X.ravel())
    print ("Y1: ", Y1.Y.shape, Y1.Y.ravel())
    print ("Y2: ", Y2.Y.shape, Y2.Y.ravel())
    print ("Y3: ", Y3.Y.shape, Y3.Y.ravel())

    print ("dX: ", l.dX.shape, l.dX.ravel())

    y = np.concatenate([Y1.Y.ravel(), Y2.Y.ravel(), Y3.Y.ravel()])
    assert (X.ravel() == y).all()
    assert (l.dX.ravel() == np.arange(l.dX.size)).all()

def test_slice2():
    def go_slice1(axis):
        X = np.arange(6*7*8*9).reshape((6,7,8,9))
        slice_point = 5
        l = L.Slice(X, axis = axis, slice_point = slice_point)
        y1, y2 = l
        l.reshape()
        l.forward()
        axes1 = [slice(None)] * 4
        axes2 = [slice(None)] * 4
        axes1[axis] = slice(None, slice_point) 
        axes2[axis] = slice(slice_point, None) 
        assert np.allclose(y1.Y, X[axes1])
        assert np.allclose(y2.Y, X[axes2])

        y1.dY = np.random.random(y1.Y.shape)
        y2.dY = np.random.random(y2.Y.shape)
        l.backward()
        assert np.allclose(l.dX, np.concatenate([y1.dY, y2.dY], axis = axis))
    def go_slice2(axis):
        X = np.arange(6*7*8*9).reshape((6,7,8,9))
        l = L.Slice(X, axis = axis, slice_points = [3,5])
        y1,y2,y3 = l
        l.reshape()
        l.forward()
        axes1 = [slice(None)] * 4
        axes2 = [slice(None)] * 4
        axes3 = [slice(None)] * 4
        axes1[axis] = slice(None, 3) 
        axes2[axis] = slice(3, 5) 
        axes3[axis] = slice(5, None) 
        assert np.allclose(y1.Y, X[axes1])
        assert np.allclose(y2.Y, X[axes2])
        assert np.allclose(y3.Y, X[axes3])


        y1.dY = np.random.random(y1.Y.shape)
        y2.dY = np.random.random(y2.Y.shape)
        y3.dY = np.random.random(y3.Y.shape)
        l.backward()
        assert np.allclose(l.dX, np.concatenate([y1.dY, y2.dY, y3.dY], axis = axis))
    for axis in range(4):
        go_slice1(axis = axis)
        go_slice2(axis = axis)
