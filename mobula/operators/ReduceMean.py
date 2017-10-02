from .Layer import *

class ReduceMean(Layer):
    def __init__(self, model, *args, **kwargs):
        Layer.__init__(self, model, *args, **kwargs)
        self.axis = kwargs.get("axis", 1)
        if type(self.axis) == list:
            self.axis = tuple(set(self.axis))
        else:
            self.axis = (self.axis, )
    def reshape(self):
        self.reduce_s = [1] * self.X.ndim
        self.reduce_n = 1
        self.new_shp = list(self.X.shape)
        for i in self.axis:
            u = self.X.shape[i]
            self.reduce_s[i] = u 
            self.new_shp[i] = 1
            self.reduce_n *= u 
        y_shp = [s for i, s in enumerate(self.X.shape) if i not in self.axis]
        self.Y = np.zeros(y_shp)
    def forward(self):
        self.Y = np.mean(self.X, axis = self.axis)
    def backward(self):
        self.dX = np.tile(self.dY.reshape(self.new_shp) / self.reduce_n, self.reduce_s)
