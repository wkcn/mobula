from .Layer import *
from ..Net import * 

class Dropout(Layer):
    def __init__(self, model, *args, **kwargs):
        Layer.__init__(self, model, *args, **kwargs)
        self.ratio = kwargs["ratio"]
        self.scale = 1.0 / (1.0 - self.ratio)
    def __str__(self):
        return "It is a Dropout Layer"
    def reshape(self):
        self.Y = np.zeros(self.X.shape)
    def forward(self):
        if Net.PHASE == Net.TRAIN:
            self.mask = (np.random.random(self.X.shape) > self.ratio)
            self.Y = np.multiply(self.X, self.mask) * self.scale
        else:
            self.Y = self.X
    def backward(self):
        if Net.PHASE == Net.TRAIN:
            self.dX = np.multiply(self.dY, self.mask) * self.scale
        else:
            self.dX = self.dY

