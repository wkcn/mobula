from .Layer import *

class Power(Layer):
    def __init__(self, models, *args, **kwargs):
        Layer.__init__(self, models, *args, **kwargs)
        self.n = kwargs.get("n", 1)
    def reshape(self):
        self.Y = np.zeros(self.X.shape)
    def forward(self):
        self.Y = np.power(self.X, self.n)
    def backward(self):
        self.dX = self.dY * np.power(self.X, self.n - 1) * self.n 
