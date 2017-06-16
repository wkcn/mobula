from .Layer import *

class PReLU(Layer):
    def __init__(self, model, *args, **kwargs):
        Layer.__init__(self, model, *args, **kwargs)
        self.a = 0.25
    def __str__(self):
        return "It is a PReLU Layer"
    def reshape(self):
        self.Y = np.zeros(self.X.shape)
    def reshape2(self):
        self.dX = np.zeros(self.X.shape)
    def forward(self):
        self.Y = self.X.copy()
        self.Y[self.X < 0] *= self.a 
    def backward(self):
        self.dX = self.dY.copy()
        self.dX[self.X < 0] *= self.a
    def update(self, lr):
        bx = self.X < 0
        if bx.any():
            self.a -= lr * np.mean(np.multiply(self.dY[bx], self.X[bx]))
