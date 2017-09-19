from .Layer import *

class Positive(Layer):
    def __init__(self, model, *args, **kwargs):
        Layer.__init__(self, model, *args, **kwargs)
    def reshape(self):
        self.Y = self.X 
    def forward(self):
        self.Y = self.X 
    def backward(self):
        self.dX = self.dY
