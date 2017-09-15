import mobula.operators as O
import numpy as np

def test_power():
    N, C, H, W = 1,2,1,3
    a = np.random.random((N, C, H, W))
    for n in range(4):
        l = O.Power(a, n = n)
        y = l.eval()
        l.dY = np.random.random(l.Y.shape)
        l.backward()
        assert np.allclose(y, np.power(a, n))
        assert np.allclose(l.dX, n * np.power(a, n-1)*l.dY)
