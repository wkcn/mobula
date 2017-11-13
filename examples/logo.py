from defines import *
import mobula
import mobula.layers as L
import mobula.solvers as S
from scipy.misc import imread, imresize, imsave
import matplotlib.pyplot as plt
import numpy as np

target_size = (42, 42)
im = imread("./mobula.png")

# Reshape
im = imresize(im, target_size)

# TO GRAY
im = im[:,:,0] * 0.299 + im[:,:,1] * 0.587 + im[:,:,2] * 0.114
h, w = im.shape

t = 1
Y = im.reshape((1, h, w, t)).transpose((0, 3, 1, 2))
X = np.random.random((1, t, h, w)) - 0.5

data, label = L.Data([X, Y])
conv = L.Conv(data, dim_out = 42, kernel = 3, pad = 1)
relu = L.ReLU(conv)
convt = L.ConvT(relu, dim_out = t, kernel = 3, pad = 1)
relu2 = L.ReLU(convt)
loss = L.MSE(relu2, label = label)

# Net Instance
net = mobula.Net()
# Set Loss Layer
net.set_loss(loss)
# Set Solver
net.set_solver(S.Momentum())

# Learning Rate
net.lr = 2e-6 

start_iter = 0 
max_iter = 10000
plt.ion()
for i in range(start_iter, max_iter + 1):
    net.forward()
    net.backward()

    if i % 100 == 0:
        print ("Iter: %d, Cost: %f" % (i, loss.loss))
        net.time()
        if i % 100 == 0:
            im = relu2.Y.reshape((h,w))
            plt.title("Iter: %d" % i)
            plt.imshow(im, "gray")
            plt.show()
            plt.pause(0.001)

plt.ioff()
plt.show()
