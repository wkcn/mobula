from .Layer import *

class Crop(Layer):
    def __init__(self, models, *args, **kwargs):
        # the type of models is list
        Layer.__init__(self, models[0], *args, **kwargs)
        self.R = models[1]
        self.axis = kwargs.get("axis", 2) # N,C,H,W
        self.offset = kwargs.get("offset", 0) # int or list
    def reshape(self):
        self.Y = np.zeros(self.R.shape)
        n = self.X.ndim - self.axis
        if type(self.offset) == int:
            self.axes = [slice(None)] * self.axis + [slice(self.offset, self.offset + self.R.Y.shape[i + self.axis]) for i in range(self.R.Y.ndim - self.axis)]
        else:
            assert len(self.offset) == n, "len(Crop.offset) must be the number of dims which will be cropped"
            self.axes = [slice(None)] * self.axis + [slice(o, o + self.R.Y.shape[i + self.axis]) for i, o in enumerate(self.offset)]
    def forward(self):
        self.Y = self.X[self.axes]
    def backward(self):
        self.dX = np.zeros(self.X.shape)
        self.dX[self.axes] = self.dY
