from mobula import Net
from mobula.layers import Data, FC

data = Data()
net = Net()

fc1 = FC(data, "fc1")
fc2 = FC(fc1, "fc2")
fc3 = FC(fc2, "fc3")

print (fc3)
