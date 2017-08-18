from .Layer import *

class Crop(Layer):
    def __init__(self, models, *args, **kwargs):
        # the type of models is list
        Layer.__init__(self, models[0], *args, **kwargs)
        self.R = models[1]
        self.axis = kwargs.get("axis", 2)
        assert 0 <= self.axis < 4
        self.offset = kwargs.get("offset", 0)
        if type(self.offset) != list:
            self.offset = np.ones(4 - self.axis) * self.offset
        else:
            assert len(self.offset) == 4 - self.axis, RuntimeError("len(Crop.offset) must be 4 - self.axis")
        self.offset = np.array(self.offset).astype(np.int)
    def reshape(self):
        self.Y = np.zeros(self.R.shape)
    def forward(self):
        s = self.R.shape
        o = self.offset
        if self.axis == 0:
            self.Y = self.X[o[0]:o[0]+s[0], o[1]:o[1]+s[1], o[2]:o[2]+s[2], o[3]:o[3]+s[3]] 
        elif self.axis == 1:
            self.Y = self.X[:, o[0]:o[0]+s[1], o[1]:o[1]+s[2], o[2]:o[2]+s[3]] 
        elif self.axis == 2:
            self.Y = self.X[:, :, o[0]:o[0]+s[2], o[1]:o[1]+s[3]] 
        else:
            self.Y = self.X[:, :, :, o[0]:o[0]+s[3]] 
    def backward(self):
        s = self.R.shape
        o = self.offset
        self.dX = np.zeros(self.X.shape)
        if self.axis == 0:
            self.dX[o[0]:o[0]+s[0], o[1]:o[1]+s[1], o[2]:o[2]+s[2], o[3]:o[3]+s[3]] = self.dY
        elif self.axis == 1:
            self.dX[:, o[0]:o[0]+s[1], o[1]:o[1]+s[2], o[2]:o[2]+s[3]] = self.dY
        elif self.axis == 2:
            self.dX[:, :, o[0]:o[0]+s[2], o[1]:o[1]+s[3]] = self.dY
        else:
            self.dX[:, :, :, o[0]:o[0]+s[3]] = self.dY
