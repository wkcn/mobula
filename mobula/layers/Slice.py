from .Layer import *

class Slice(Layer):
    def __init__(self, model, *args, **kwargs):
        Layer.__init__(self, model, *args, **kwargs)
        self.axis = kwargs.get("axis", 1)
        if "slice_point" in kwargs:
            self.slice_points = [kwargs["slice_point"]]
        else:
            self.slice_points = kwargs.get("slice_points", [])
        self.set_output(len(self.slice_points) + 1)
    def reshape(self):
        self.slices = [[slice(None)] * self.X.ndim for _ in range(len(self.slice_points) + 1)]
        last = None
        for i, k in enumerate(self.slice_points):
            s = slice(last, k)
            self.slices[i][self.axis] = s
            last = k
            self.Y[i] = self.X[s]
        s = slice(last, None)
        self.slices[-1][self.axis] = s
        self.Y[-1] = self.X[s] 
    def forward(self):
        self.Y = [self.X[s] for s in self.slices]
    def backward(self):
        self.dX = np.concatenate(self.dY, axis = self.axis)
