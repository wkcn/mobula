from .Layer import *

class Abs(Layer):
    def __init__(self, models, *args, **kwargs):
        Layer.__init__(self, models, *args, **kwargs)
    def reshape(self):
        self.Y = np.zeros(self.X.shape)
    def forward(self):
        self.Y = np.abs(self.X)
    def backward(self):
        self.dX = np.zeros(self.X.shape) 
        self.dX[self.X > 0] = 1.0 
        self.dX[self.X < 0] = -1.0
        self.dX *= self.dY
