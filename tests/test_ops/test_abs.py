import mobula as M
import mobula.operators as O
import numpy as np

def test_abs():
    X = np.random.random((3,4,5)) - 0.5
    l = M.abs(X)
    assert np.allclose(l.eval(), np.abs(X))
    l.dY = np.random.random(l.Y.shape)
    l.backward()
    dX = np.zeros(X.shape)
    dX[X > 0] = l.dY[X > 0]
    dX[X < 0] = -l.dY[X < 0]
    assert np.allclose(dX, l.dX)
