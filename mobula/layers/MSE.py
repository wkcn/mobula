from .Layer import *

class MSE(Layer):
    def __init__(self, model, *args, **kwargs):
        Layer.__init__(self, model, *args, **kwargs)
        self.label = kwargs["label"] # (N, C)
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
