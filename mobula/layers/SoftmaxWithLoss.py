from .LossLayer import *

class SoftmaxWithLoss(LossLayer):
    def __init__(self, model, *args, **kwargs):
        LossLayer.__init__(self, model, *args, **kwargs)
    def __str__(self):
        return "It is a SoftmaxWithLoss Layer"
    def reshape(self):
        self.Y = np.zeros(self.X.shape)
        n = self.Y.shape[0]
        self.a = np.arange(n)
    def forward(self):
        e = np.exp(self.X)
        self.s = np.sum(e, 1)
        self.Y = np.asarray(e / np.asmatrix(self.s).T)
    def backward(self):
        self.dX = self.Y.copy()
        self.dX[self.a, self.label.ravel()] -= 1.0
    @property
    def loss(self):
        return np.mean(np.log(self.s)) - np.mean(self.X[self.a, self.label.ravel()]) 
