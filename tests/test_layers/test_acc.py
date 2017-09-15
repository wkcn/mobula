import mobula.layers as L
import numpy as np

def test_acc():
    X = np.array([[0, 1, 2],
                    [1, 2, 0],
                    [0, 1, 2],
                    [1, 2, 0]])

    Y = np.array([1, 0, 2, 1]).reshape((-1, 1)) 
    # top-k
    # 1            [False, False, True, True]
    # 2            [True, True, True, True]

    target = [np.array([False, False, True, True]), np.array([True, True, True, True])]

    [data, label] = L.Data([X, Y])
    for i in range(2):
        l = L.Accuracy(data, label = label, top_k = 1 + i)
        l.reshape()
        l.forward()
        assert l.Y == np.mean(target[i])
