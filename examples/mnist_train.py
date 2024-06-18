from defines import *
import mobula
import mobula.layers as L
import mobula.solvers as S
import os
from LeNet5 import *

INPUT_FILE = "./train.csv"
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
X = data[:, 1:]
labels = data[:, 0].astype(np.int32)
# one-hot
#Y = np.eye(10)[labels.ravel()] 
print ("Read OK", n)

Xmean = np.mean(X, 0)
np.save("xmean.npy", Xmean)

# Subtract mean and normalize
X = (X - Xmean) / 255.0

# transfer the shape of X to NCHW
# N, C, H, W = n, 1, 28, 28
X.resize((n, 1, 28, 28))

# Get LeNet5
nn = LeNet5(X, labels)
net = nn.net

# Set Solver
net.set_solver(S.Momentum())

# Learning Rate
net.lr = 0.005

'''
If start_iter > 0, load the existed model and continue to train.
Otherwise, initialize weights and start to train. 
'''

start_iter = 0 
max_iter = 100000
filename = RESULT_PATH + "/kaggle%d.net"
if start_iter > 0:
    # Load the weights from the existed model
    net.load(filename % start_iter)

for i in range(start_iter, max_iter):
    net.forward()
    net.backward()

    if i % 100 == 0:
        print ("Iter: %d, Cost: %f" % (i, nn.loss))
        net.time()

        # test 30 iteration
        vs = []
        for u in range(30):
            net.forward()
            pre = np.argmax(nn.Y,1).ravel()
            ra = nn.label.ravel()
            if u % 10 == 0: 
                print ((pre, ra))
            bs = (pre == ra) 
            b = np.mean(bs)
            vs.append(b)
        acc = np.mean(vs)
        print ("Accuracy: %f" % (acc))
        if i % 200 == 0 and acc > 0.95:
            net.save(filename % i)

print ("Over :-)")
