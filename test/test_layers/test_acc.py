from test_layers import *
import mobula.layers as L

X = np.array([[0, 1, 2],
                [1, 2, 0],
                [0, 1, 2],
                [1, 2, 0]])

Y = np.array([1, 0, 2, 1]).reshape((-1, 1)) 
# top-k
# 1            [False, False, True, True]
# 2            [True, True, True, True]

[data, label] = L.Data([X, Y], "data")()
l = L.Accuracy(data, "acc", label = label, top_k = 2)

l.reshape()
l.forward()
print (l.Y)
