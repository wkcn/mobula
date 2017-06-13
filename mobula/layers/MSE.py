from .LossLayer import *

class MSE(LossLayer):
    def __init__(self, model, *args, **kwargs):
        LossLayer.__init__(self, model, *args, **kwargs)
    def __str__(self):
        return "It is a Mean Squared Error Layer"
    def reshape(self):
        self.Y = 0.
    def forward(self):
        self.d = (self.X - self.label)
        self.Y = np.mean(np.multiply(self.d,self.d))
    def reshape2(self):
        self.dX = np.zeros(self.X.shape) 
    def backward(self):
        self.dX = 2 * self.d
    @property
    def loss(self):
        return self.Y
