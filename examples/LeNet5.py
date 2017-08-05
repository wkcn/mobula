from defines import *
import numpy as np
import mobula
import mobula.layers as L

class LeNet5:
    def __init__(self, X, labels):

        data, label = L.Data([X, labels], "data", batch_size = 100)()
        conv1 = L.Conv(data, "conv1", dim_out = 20, kernel = 5)
        pool1 = L.Pool(conv1, "pool1", pool = L.Pool.MAX, kernel = 2, stride = 2)
        conv2 = L.Conv(pool1, "conv2", dim_out = 50, kernel = 5)
        pool2 = L.Pool(conv2, "pool2", pool = L.Pool.MAX, kernel = 2, stride = 2)
        fc3   = L.FC(pool2, "fc3", dim_out = 500)
        relu3 = L.ReLU(fc3, "relu3")
        pred  = L.FC(relu3, "pred", dim_out = 10)
        loss = L.SoftmaxWithLoss(pred, "loss", label = label)

        # Net Instance
        self.net = mobula.Net()

        # Set Loss Layer
        self.net.set_loss(loss)

    @property
    def Y(self):
        return self.net["pred"].Y

    @property
    def loss(self):
        return self.net["loss"].loss

    @property
    def label(self):
        return self.net["data"](1).Y
