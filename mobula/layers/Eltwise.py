from .Layer import *

# Elementwise Operations
class Eltwise(Layer):
    SUM, PROD, MAX = range(3)
    def __init__(self, models, *args, **kwargs):
        # the type of models is list
        Layer.__init__(self, models, *args, **kwargs)
        if "coeffs" in kwargs:
            self.coeffs = np.array(kwargs["coeffs"])
            if self.coeffs.size != len(self.X):
                raise ValueError("The number of coeffs must be the number of inputs :-(")
        else:
            self.coeffs = np.ones(len(self.X))
        self.op = kwargs.get("op", Eltwise.SUM)
        if self.op < 0 or self.op > 3:
            raise RuntimeError("Eltwise operation Error: ", self.op)
    def reshape(self):
        self.Y = np.zeros(self.X[0].shape)
    def forward(self):
        if self.op == Eltwise.SUM:
            self.Y = 0
            for i in range(len(self.X)):
                self.Y += self.X[i] * self.coeffs[i]
        elif self.op == Eltwise.PROD:
            for i in range(len(self.X)):
                x = self.X[i]
                x[x == 0] = 1e-100
            self.Y = np.prod(self.X, 0) * np.prod(self.coeffs)
        else:
            # self.op == Eltwise.MAX
            cy =  [self.X[i] * self.coeffs[i] for i in range(len(self.X))]
            self.argmax = np.argmax(cy, 0)
            self.Y = np.max(cy, 0)
    def backward(self):
        if self.op == Eltwise.SUM:
            for i in range(len(self.X)):
                if self.coeffs[i] == 1.0:
                    self.dX[i] = self.dY
                else:
                    self.dX[i] = self.coeffs[i] * self.dY
        elif self.op == Eltwise.PROD:
            for i in range(len(self.X)):
                self.dX[i] = self.Y / self.X[i] * self.dY
        else:
            # self.op == Eltwise.MAX
            for i in range(len(self.X)):
                self.dX[i] = np.zeros(self.X[0].shape)
                b = (self.argmax == i)
                self.dX[i][b] = self.coeffs[i] * self.dY[b]
