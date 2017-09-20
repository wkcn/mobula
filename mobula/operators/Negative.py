from .Layer import *

class Negative(Layer):
    def __init__(self, model, *args, **kwargs):
        self.check_inputs(model, 1)
        Layer.__init__(self, model, *args, **kwargs)
    def reshape(self):
        self.Y = np.zeros(self.X.shape)
    def forward(self):
        self.Y = -self.X 
    def backward(self):
        self.dX = -self.dY
