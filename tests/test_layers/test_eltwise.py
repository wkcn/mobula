import mobula
import mobula.layers as L
import numpy as np

def go_eltwise(op):
    a = np.array([1,0,6]).astype(np.float)
    b = np.array([4,5,3]).astype(np.float)
    print ("a: ", a)
    print ("b: ", b)
    data1 = L.Data(a)
    data2 = L.Data(b)
    coeffs = np.array([-1.0,1.2])
    l = L.Eltwise([data1,data2], op = op, coeffs = coeffs)
    l.reshape()
    l.forward()
    print ("Y: ", l.Y)
    dY = np.array([7, 8, 9]).astype(np.float)
    l.dY = dY 
    print ("dY: ", l.dY)
    l.backward()
    print ("dX: ", l.dX[0], l.dX[1])
    c0, c1 = coeffs
    if op == L.Eltwise.SUM:
        Y = c0 * a + c1 * b
        dX0 = c0 * dY 
        dX1 = c1 * dY 
    elif op == L.Eltwise.PROD:
        Y = a * b * c0 * c1
        dX0 = b * dY * c0 * c1 
        dX1 = a * dY * c0 * c1 
    elif op == L.Eltwise.MAX:
        Y = np.max([c0*a,c1*b], 0)
        i = np.argmax([c0*a,c1*b], 0)
        dX0 = np.zeros(a.shape)
        dX1 = np.zeros(b.shape)
        dX0[i == 0] = dY[i == 0] * c0
        dX1[i == 1] = dY[i == 1] * c1

    print ("Y", l.Y, Y)
    assert np.allclose(l.Y, Y)
    assert np.allclose(l.dX[0], dX0)
    assert np.allclose(l.dX[1], dX1)

def test_eltwise():
    print ("TEST SUM")
    go_eltwise(L.Eltwise.SUM)

    print ("TEST PROD")
    go_eltwise(L.Eltwise.PROD)

    print ("TEST MAX")
    go_eltwise(L.Eltwise.MAX)
