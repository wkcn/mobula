from testlib import *
import mobula
import mobula.layers as L
import mobula.solvers as S
import os
from LeNet5 import *

INPUT_FILE = "./test.csv"
RESULT_PATH = "./mnist_kaggle"

# Create the directory if it doesn't exists
if not os.path.exists(RESULT_PATH):
    os.mkdir(RESULT_PATH)

# Load Data
try:
    fin = open(INPUT_FILE)
except:
    print ("%s doesn't exist. Please download the dataset on Kaggle: https://www.kaggle.com/c/digit-recognizer" % INPUT_FILE)
    import sys
    sys.exit()

data = np.loadtxt(fin, delimiter = ",", skiprows = 1)
fin.close()

n = len(data)
X = data
print ("Read OK", n)

Xmean = np.load("xmean.npy")

# Subtract mean and normalize
X = (X - Xmean) / 255.0

# labels as index for predicting
labels = np.arange(n)

# transfer the shape of X to NCHW
# N, C, H, W = n, 1, 28, 28
X.resize((n, 1, 28, 28))

# Get LeNet5
nn = LeNet5(X, labels)
net = nn.net
data_layer = net["data"]
index = data_layer(1)

'''
If iter_num > 0, load the existed model and continue to train.
Otherwise, initialize weights and start to train. 
'''

iter_num = 25200 

filename = RESULT_PATH + "/kaggle%d.net"
if iter_num > 0:
    # Load the weights from the existed model
    net.load(filename % iter_num)

ok = 0
res = [None] * n
while ok < n:
    net.forward()
    pre = np.argmax(nn.Y,1)
    for d in range(data_layer.batch_size):
        ind = index.Y[d]
        if res[ind] == None:
            ok += 1
            print ("%d / %d" % (ok, n))
        res[ind] = pre[d]
fout = open("result-%d.csv" % iter_num, "w")
fout.write("ImageId,Label\n")
for i in range(n):
    fout.write("%d,%d\n" % (i + 1, res[i]))

print ("Over :-)")
