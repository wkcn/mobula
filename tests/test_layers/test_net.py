import mobula
import mobula.layers as L
import numpy as np

def test_net():
    X = np.random.random((4,2,1,1))
    Y1 = np.random.random((4,5))
    Y2 = np.random.random((4,5))

    [x,y1,y2] = L.Data([X,Y1,Y2])()
    fc0 = L.FC(x, dim_out = 10)
    fc1 = L.FC(fc0, dim_out = 5)
    fc2 = L.FC(fc0, dim_out = 5)

    loss1 = L.MSE(fc1, label = Y1)
    loss2 = L.MSE(fc2, label = Y2)

    net = mobula.Net()
    net.set_loss(loss1 + loss2)

    net.lr = 0.5
    for i in range(10):
        net.forward()
        net.backward()
        print ("Iter: %d, Cost: %f" % (i, loss1.Y + loss2.Y))

    assert np.allclose(fc0.dY, fc1.dX + fc2.dX)
