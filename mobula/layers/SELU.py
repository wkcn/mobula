from .Layer import *

class SELU(Layer):
    def __init__(self, model, *args, **kwargs):
        Layer.__init__(self, model, *args, **kwargs)
        self.scale = kwargs.get("scale", 1.0507009873554804934193349852946)
        self.alpha = kwargs.get("alpha", 1.6732632423543772848170429916717)
    def reshape(self):
        self.Y = np.zeros(self.X.shape)
    def forward(self):
        b = self.X > 0
        self.Y[b] = self.scale * self.X[b]
        self.sae = self.scale * self.alpha * np.exp(self.X[~b])
        self.Y[~b] = self.sae - self.scale * self.alpha 
    def backward(self):
        self.dX = np.zeros(self.X.shape)
        b = self.X > 0
        self.dX[b] = self.scale
        self.dX[~b] = self.sae
        self.dX = np.multiply(self.dX, self.dY)
