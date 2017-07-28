from .Layer import *

class Dropout(Layer):
    def __init__(self, model, *args, **kwargs):
        Layer.__init__(self, model, *args, **kwargs)
        self.ratio = kwargs["ratio"]
        self.scale = 1.0 / (1.0 - self.ratio)
    def reshape(self):
        self.Y = np.zeros(self.X.shape)
    def forward(self):
        if self.is_training():
            self.mask = (np.random.random(self.X.shape) > self.ratio)
            self.Y = np.multiply(self.X, self.mask) * self.scale
        else:
            self.Y = self.X
    def backward(self):
        if self.is_training():
            self.dX = np.multiply(self.dY, self.mask) * self.scale
        else:
            self.dX = self.dY

