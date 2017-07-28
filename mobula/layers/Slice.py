from .Layer import *

# TODO: Optimize Condition Branches
class Slice(Layer):
    def __init__(self, model, *args, **kwargs):
        Layer.__init__(self, model, *args, **kwargs)
        self.axis = kwargs.get("axis", 1)
        if self.axis < 0:
            self.axis = 4 + self.axis
        assert 0 <= self.axis  < 4
        if "slice_point" in kwargs:
            self.slice_points = [kwargs["slice_point"]]
        else:
            self.slice_points = kwargs.get("slice_points", [])
        self.set_output(len(self.slice_points) + 1)
    def reshape(self):
        j = 0
        sz = list(self.X.shape)
        m = sz[self.axis]
        n = len(self.slice_points)
        for i in range(n):
            #[j, p)
            p = self.slice_points[i]
            sz[self.axis] = p - j
            self.Y[i] = np.zeros(sz)
            j = p
        sz[self.axis] = m - j
        self.Y[n] = np.zeros(sz)
    def forward(self):
        j = 0
        n = len(self.slice_points)
        if self.axis == 0:
            for i in range(n):
                #[j, p)
                p = self.slice_points[i]
                self.Y[i] = self.X[j:p]
                j = p
            self.Y[n] = self.X[j:]
        elif self.axis == 1:
            for i in range(n):
                #[j, p)
                p = self.slice_points[i]
                self.Y[i] = self.X[:, j:p]
                j = p
            self.Y[n] = self.X[:, j:]
        elif self.axis == 2:
            for i in range(n):
                #[j, p)
                p = self.slice_points[i]
                self.Y[i] = self.X[:, :, j:p]
                j = p
            self.Y[n] = self.X[:, :, j:]
        else: #if self.axis == 3:
            for i in range(n):
                #[j, p)
                p = self.slice_points[i]
                self.Y[i] = self.X[:, :, :, j:p]
                j = p
            self.Y[n] = self.X[:, :, :, j:]
    def backward(self):
        self.dX = np.concatenate(self.dY, axis = self.axis)
