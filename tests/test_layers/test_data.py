import mobula
import mobula.layers as L
import numpy as np

def test_data():
    X = np.arange(1, 11).reshape((10, 1))

    target = [[1,2,3],[4,5,6],[7,8,9],[2,3,4],[5,6,7],[8,9,10]]

    data = L.Data(X, "data", batch_size = 3)()
    data.reshape()


    for i in range(50):
        data.forward()
        assert (data.Y.ravel() == target[i % 6]).all()
