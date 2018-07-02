from .Layer import *

class PReLU(Layer):
    def __init__(self, model, *args, **kwargs):
        Layer.__init__(self, model, *args, **kwargs)
        alpha = kwargs.get("alpha", 0.25)
        self.alpha = np.array(alpha)
    def reshape(self):
        self.Y = np.zeros(self.X.shape)
    def forward(self):
        self.Y = self.X.copy()
        self.Y[self.X < 0] *= self.alpha
    def backward(self):
        self.dX = self.dY.copy()
        self.dX[self.X <= 0] *= self.alpha
    @property
    def params(self):
        return [self.alpha]
    @property
    def grads(self):
        bx = self.X < 0
        return [np.sum(np.multiply(self.dY[bx], self.X[bx]))] if bx.any() else [0.0]
