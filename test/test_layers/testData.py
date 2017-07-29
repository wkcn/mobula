from test_layers import *
import mobula
import mobula.layers as L

X = np.arange(10) + 1
print (X)
data = L.Data(X, "data", batch_size = 3)
data.reshape()
print (data.Y)

for i in range(20):
    data.forward()
    print (i+1, data.Y)
