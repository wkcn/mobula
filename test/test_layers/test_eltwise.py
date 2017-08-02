from test_layers import *
import mobula
import mobula.layers as L

def test_eltwise(op):
    a = np.array([1,0,6]).astype(np.float)
    b = np.array([4,5,3]).astype(np.float)
    print ("a: ", a)
    print ("b: ", b)
    data1 = L.Data(a, "data1")
    data2 = L.Data(b, "data1")
    coeffs = np.array([1.0,1.0])
    l = L.Eltwise([data1,data2], "eltwise", op = op, coeffs = coeffs)
    l.reshape()
    l.forward()
    print ("Y: ", l.Y)
    l.dY = np.array([7,8,9]).astype(np.float)
    print ("dY: ", l.dY)
    l.backward()
    print ("dX: ", l.dX[0], l.dX[1])

print ("TEST SUM")
test_eltwise(L.Eltwise.SUM)


print ("TEST PROD")
test_eltwise(L.Eltwise.PROD)


print ("TEST MAX")
test_eltwise(L.Eltwise.MAX)
