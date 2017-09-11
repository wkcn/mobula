from .LossLayer import *

class SoftmaxWithLoss(LossLayer):
    def __init__(self, model, *args, **kwargs):
        LossLayer.__init__(self, model, *args, **kwargs)
    def reshape(self):
        self.Y = 0.0 
        self.a = np.arange(self.X.shape[0])
    def forward(self):
        e = np.exp(self.X)
        self.s = np.sum(e, 1)
        self.Y = (np.mean(np.log(self.s)) - np.mean(self.X[self.a, self.label.ravel()]))
        self.softmax = np.asarray(e / np.asmatrix(self.s).T)
    def backward(self):
        self.dX = self.softmax.copy()
        self.dX[self.a, self.label.ravel()] -= 1.0
        self.dX *= self.dY
