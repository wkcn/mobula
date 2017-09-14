from .Layer import *

class Softmax(Layer):
    def __init__(self, model, *args, **kwargs):
        Layer.__init__(self, model, *args, **kwargs)
        self.axis = kwargs.get("axis", 1) # N,C,H,W
    def reshape(self):
        self.Y = np.zeros(self.X.shape)
    def forward(self):
        self.e = np.exp(self.X)
        s = np.sum(self.e, axis = self.axis)
        axes = [slice(None)] * 4
        axes[self.axis] = np.newaxis
        self.Y = self.e / s[axes]
    def backward(self):
        self.dX = np.multiply(self.dY, self.Y - np.square(self.Y))
