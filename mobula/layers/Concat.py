from .Layer import *

class Concat(Layer):
    def __init__(self, models, *args, **kwargs):
        Layer.__init__(self, models, *args, **kwargs)
        self.axis = kwargs.get("axis", 1)
    def reshape(self):
        last = 0
        self.slices = [[slice(None)] * self.X[0].ndim for _ in range(len(self.X))]
        for i, x in enumerate(self.X):
            ss = x.shape[self.axis]
            self.slices[i][self.axis] = slice(last, last + ss)
            last += ss

        shp = list(self.X[0].shape)
        shp[self.axis] = last
        self.Y = np.zeros(shp)
    def forward(self):
        self.Y = np.concatenate(self.X, axis = self.axis)
    def backward(self):
        self.dX = [self.dY[s] for s in self.slices]
