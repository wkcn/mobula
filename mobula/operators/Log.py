from .Layer import *

class Log(Layer):
    def __init__(self, models, *args, **kwargs):
        Layer.__init__(self, models, *args, **kwargs)
    def reshape(self):
        self.Y = np.zeros(self.X.shape)
    def forward(self):
        self.Y = np.log(self.X)
    def backward(self):
        self.dX = self.dY * (1.0 / self.X) 
