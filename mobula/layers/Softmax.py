from .Layer import *

class Softmax(Layer):
    def __init__(self, model, *args, **kwargs):
        Layer.__init__(self, model, *args, **kwargs)
    def __str__(self):
        return "It is a Softmax Layer"
    def reshape(self):
        self.Y = np.zeros(self.X.shape)
    def forward(self):
        self.e = np.exp(self.X)
        s = np.sum(self.e)
        self.Y = self.e / s
    def backward(self):
        self.dX = np.multiply(self.dY, self.e)
