from .Layer import *

class Multiply(Layer):
    def __init__(self, models, *args, **kwargs):
        assert len(models) == 2, "the number of inputs of Subtract must be 2"
        Layer.__init__(self, models, *args, **kwargs)
    def reshape(self):
        self.Y = np.zeros(self.X[0].shape)
    def forward(self):
        self.Y = np.multiply(self.X[0], self.X[1]) 
    def backward(self):
        self.dX = [np.multiply(self.dY, self.X[1]), np.multiply(self.dY, self.X[0])]

class MultiplyConstant(Layer):
    def __init__(self, model, *args, **kwargs):
        Layer.__init__(self, model, *args, **kwargs)
        self.constant = kwargs["constant"]
    def reshape(self):
        self.Y = np.zeros(self.X.shape)
    def forward(self):
        self.Y = self.X * self.constant
    def backward(self):
        self.dX = self.dY * self.constant 

Multiply.OP_L = MultiplyConstant
Multiply.OP_R = MultiplyConstant
