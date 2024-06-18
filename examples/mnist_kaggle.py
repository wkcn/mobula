from defines import *
import mobula
import mobula.layers as L
import mobula.solvers as S
import os
import sys

def print_info():
    print ('''
* mnist_kaggle.py
Please download the dataset on Kaggle: https://www.kaggle.com/c/digit-recognizer
Then put train.csv and test.csv in the directory which mnist_kaggle.py is in. 
Usage:
    Recommend using Anaconda for better performance.
    Train:
        python mnist_kaggle.py train [MODEL] 
    Test:
        python mnist_kaggle.py test  MODEL

    MODEL:  the file saving the weights of the network
    '''
    )


traning = True
model_file = None

'''
if len(sys.argv) == 1:
    print_info()
    sys.exit()
else:
    phase = sys.argv[1].lower() 
    if phase == "train":
        traing = True
        if len(sys.argv) >= 3:
            model_file = sys.argv[2]
    elif phase == "test":
        traing = False
        if len(sys.argv) >= 3:
            model_file = sys.argv[2]
        else:
            print_info()
            sys.exit()
    else:
        print_info()
        sys.exit()
'''


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

# LeNet-5
data, label = L.Data([X, labels], "data", batch_size = 100)
conv1 = L.Conv(data, "conv1", dim_out = 20, kernel = 5)
pool1 = L.Pool(conv1, "pool1", pool = L.Pool.MAX, kernel = 2, stride = 2)
relu1 = L.ReLU(pool1, "relu1")
conv2 = L.Conv(relu1, "conv2", dim_out = 50, kernel = 5)
pool2 = L.Pool(conv2, "pool2", pool = L.Pool.MAX, kernel = 2, stride = 2)
relu2 = L.ReLU(pool2, "relu2")
fc3   = L.FC(relu2, "fc3", dim_out = 500)
relu3 = L.ReLU(fc3, "relu3")
pred  = L.FC(relu3, "pred", dim_out = 10)
loss = L.SoftmaxWithLoss(pred, "loss", label = label)

# Net Instance
net = mobula.Net()

# Set Loss Layer
net.set_loss(loss)

# Set Solver
solver = S.Momentum(gamma = 0.1, stepsize = 1000)
solver.lr_policy = S.LR_POLICY.STEP
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
        print ("Iter: %d, Cost: %f" % (i, loss.loss))
        net.time()

        # test 30 iteration
        vs = []
        for u in range(30):
            net.forward()
            pre = np.argmax(pred.Y,1).ravel()
            ra = label.Y.ravel()
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
