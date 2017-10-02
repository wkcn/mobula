import mobula
import mobula.layers as L
import numpy as np

def test_data():
    X = np.arange(1, 11).reshape((10, 1))

    target = [[1,2,3],[4,5,6],[7,8,9],[2,3,4],[5,6,7],[8,9,10]]

    def go_data(data):
        data.reshape()

        for i in range(50):
            data.forward()
            assert (data.Y.ravel() == target[i % 6]).all()

    data = L.Data(X, batch_size = 3)
    assert data.batch_size == 3
    go_data(data)
    data = L.Data(X, batch_size = 3)()
    go_data(data)
    [data] = L.Data(X, batch_size = 3)
    go_data(data)
    data = L.Data(X, batch_size = 3)(0)
    go_data(data)
    data = L.Data([X, X.copy()], batch_size = 3)(1)
    go_data(data)

def test_data_args():
    X = np.random.random((10, 3))
    Y = np.random.random((10, 3))
    d1 = L.Data([X,Y], "data1")
    lx1, ly1 = d1()
    d2 = L.Data(X, Y, "data2")
    lx2, ly2 = d2()
    assert np.allclose(lx1.Y, lx2.Y)
    assert np.allclose(ly1.Y, ly2.Y)
