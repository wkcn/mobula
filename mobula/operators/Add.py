from .Layer import *

class Add(Layer):
    def __init__(self, models, *args, **kwargs):
        Layer.__init__(self, models, *args, **kwargs)
    def reshape(self):
        self.Y = np.zeros(self.X[0].shape)
    def forward(self):
        self.Y = np.sum(self.X, 0) 
    def backward(self):
        self.dX = [self.dY for _ in range(len(self.X))]
