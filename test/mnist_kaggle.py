from testlib import *
import mobula
import mobula.layers as L
import mobula.solvers as S
import os

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
labels = data[:, 0].astype(np.int)
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

# LeNet-5
data = L.Data(X, "Data", batch_size = 100, label = labels)
conv1 = L.Conv(data, "Conv1", dim_out = 20, kernel = 5)
pool1 = L.Pool(conv1, "pool1", pool = L.Pool.MAX, kernel = 2, stride = 2)
conv2 = L.Conv(pool1, "Conv2", dim_out = 50, kernel = 5)
pool2 = L.Pool(conv2, "pool2", pool = L.Pool.MAX, kernel = 2, stride = 2)
fc3   = L.FC(pool2, "fc3", dim_out = 500)
relu3 = L.ReLU(fc3, "relu3")
pred  = L.FC(relu3, "pred", dim_out = 10)
loss = L.SoftmaxWithLoss(pred, "loss", label_data = data)

# Net Instance
net = mobula.Net()

# Set Loss Layer
net.set_loss(loss)

# Set Solver
net.set_solver(S.Momentum())

# Learning Rate
net.lr = 0.2

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
        print ("Iter: %d, Cost: %f" % (i, loss.loss))
        net.time()

        # test 30 iteration
        vs = []
        for u in range(30):
            net.forward()
            pre = np.argmax(pred.Y,1).ravel()
            ra = data.label.ravel()
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
