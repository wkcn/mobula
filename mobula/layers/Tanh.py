from .Layer import *

class Tanh(Layer):
    def __init__(self, model, *args, **kwargs):
        Layer.__init__(self, model, *args, **kwargs)
    def __str__(self):
        return "It is a Tanh Layer"
    def reshape(self):
        self.Y = np.zeros(self.X.shape)
    def reshape2(self):
        self.dX = np.zeros(self.X.shape)
    def forward(self):
        '''
        a = np.exp(self.X)
        b = np.exp(-self.X)
        self.c = a + b 
        self.Y = (a - b) / (self.c)
        '''
        self.Y = 2.0 / (1.0 + np.exp(-2.0 * self.X)) - 1.0
    def backward(self):
        #self.dX = np.multiply(self.dY, 4.0 / np.square(self.c))
        self.dX = np.multiply(self.dY, 1.0 - np.square(self.Y))
