from .Layer import *

# It's a test layer for multi input
# MergeTest([x0, x1])

class MergeTest(Layer):
    def __init__(self, models, *args, **kwargs):
        # the type of models is list
        Layer.__init__(self, models, *args, **kwargs)
    def reshape(self):
        siz = list(self.X[0].shape)
        siz[0] += self.X[1].shape[0]
        self.Y = np.zeros(siz)
    def forward(self):
        self.Y[0::2] = self.X[0]
        self.Y[1::2] = self.X[1]
    def backward(self):
        self.dX[0] = self.dY[0::2] 
        self.dX[1] = self.dY[1::2] 
