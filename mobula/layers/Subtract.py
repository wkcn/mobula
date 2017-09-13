from .Layer import *

class Subtract(Layer):
    def __init__(self, models, *args, **kwargs):
        assert len(models) == 2, "the number of inputs of Subtract must be 2"
        Layer.__init__(self, models, *args, **kwargs)
    def reshape(self):
        self.Y = np.zeros(self.X[0].shape)
    def forward(self):
        self.Y = self.X[0] - self.X[1] 
    def backward(self):
        self.dX = [self.dY, -self.dY]
