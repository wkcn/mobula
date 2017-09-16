from .Layer import *

class PReLU(Layer):
    def __init__(self, model, *args, **kwargs):
        Layer.__init__(self, model, *args, **kwargs)
        self.a = np.array(0.25)
    def reshape(self):
        self.Y = np.zeros(self.X.shape)
    def forward(self):
        self.Y = self.X.copy()
        self.Y[self.X < 0] *= self.a
    def backward(self):
        self.dX = self.dY.copy()
        self.dX[self.X <= 0] *= self.a
    @property
    def params(self):
        return [self.a]
    @property
    def grads(self):
        bx = self.X < 0
        if bx.any():
            return [np.mean(np.multiply(self.dY[bx], self.X[bx]))]
        return [0.0]
