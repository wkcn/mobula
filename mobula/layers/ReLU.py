from .Layer import *

class ReLU(Layer):
    def __init__(self, model, *args, **kwargs):
        Layer.__init__(self, model, *args, **kwargs)
    def reshape(self):
        self.Y = np.zeros(self.X.shape)
    def forward(self):
        self.Y = self.X.copy()
        self.Y[self.X <= 0] = 0.0
    def backward(self):
        self.dX = self.dY.copy()
        self.dX[self.X <= 0] = 0.0
