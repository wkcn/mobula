from .Layer import *

class ReduceMin(Layer):
    def __init__(self, model, *args, **kwargs):
        Layer.__init__(self, model, *args, **kwargs)
        self.axis = kwargs.get("axis", 1)
    def reshape(self):
        new_shp = [k for i,k in enumerate(self.X.shape) if i != self.axis]
        self.Y = np.zeros(new_shp)
    def forward(self):
        self.idx = get_idx_from_arg(self.X, np.argmin(self.X, self.axis), self.axis)
        self.Y = get_val_from_idx(self.X, self.idx).reshape(self.Y.shape)
    def backward(self):
        self.dX = np.zeros(self.X.shape)
        self.dX.ravel()[self.idx] = self.dY.ravel()
