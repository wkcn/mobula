from .Layer import *

class Tanh(Layer):
    def __init__(self, model, *args, **kwargs):
        Layer.__init__(self, model, *args, **kwargs)
    def reshape(self):
        self.Y = np.zeros(self.X.shape)
    def forward(self):
        self.Y = 2.0 / (1.0 + np.exp(-2.0 * self.X)) - 1.0
    def backward(self):
        self.dX = np.multiply(self.dY, 1.0 - np.square(self.Y))
