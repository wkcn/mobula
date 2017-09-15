from .Layer import *

class Exp(Layer):
    def __init__(self, models, *args, **kwargs):
        Layer.__init__(self, models, *args, **kwargs)
    def reshape(self):
        self.Y = np.zeros(self.X.shape)
    def forward(self):
        self.Y = np.exp(self.X)
    def backward(self):
        self.dX = self.dY * self.Y 
