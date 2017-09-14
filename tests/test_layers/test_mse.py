import mobula.layers as L
import numpy as np

def test_mse():
    N, C, H, W = 2,3,4,5
    a = np.random.random((N, C, H, W)) - 0.5
    b = np.random.random((N, C, H, W)) - 0.5

    l = L.MSE(a, label = b)
    y = l.eval()
    d = a - b
    assert np.allclose(np.mean(np.square(d)), l.Y)
    l.dY = np.random.random(l.Y.shape)
    l.backward()
    assert np.allclose(l.dX, 2 * d * l.dY)
