from .Layer import *

class Concat(Layer):
    def __init__(self, models, *args, **kwargs):
        Layer.__init__(self, models, *args, **kwargs)
        self.axis = kwargs.get("axis", 0)
        assert 0 <= self.axis  < 4
    def reshape(self):
        sz = list(self.X[0].shape)
        self.w = np.array([x.shape[self.axis] for x in self.X])
        sz[self.axis] = np.sum(self.w)
        self.Y = np.zeros(sz)
    def forward(self):
        self.Y = np.concatenate(self.X, axis = self.axis)
    def backward(self):
        n = len(self.X)
        j = 0
        if self.axis == 0:
            for i in range(n):
                e = j + self.w[i]
                self.dX[i] = self.dY[j:e]
                j = e 
        elif self.axis == 1:
            for i in range(n):
                e = j + self.w[i]
                self.dX[i] = self.dY[:, j:e]
                j = e 
        elif self.axis == 2:
            for i in range(n):
                e = j + self.w[i]
                self.dX[i] = self.dY[:, :, j:e]
                j = e 
        else: #if self.axis == 2:
            for i in range(n):
                e = j + self.w[i]
                self.dX[i] = self.dY[:, :, :, j:e]
                j = e 
