from .Layer import *

class Reshape(Layer):
    def __init__(self, model, *args, **kwargs):
        Layer.__init__(self, model, *args, **kwargs)
        self.dims = kwargs.get("dims", np.array([-1, -1, -1, -1]))
    def reshape(self):
        xshp = self.X.shape
        self.yshp = list(xshp)
        k = None
        v = 1
        for i, d in enumerate(self.__dims):
            if d == -1:
                k = i
            else:
                if d > 0:
                    self.yshp[i] = d
                v *= self.yshp[i]
        if k is not None:
            self.yshp[k] = self.X.size // v
        self.Y = self.X.reshape(self.yshp)
    def forward(self):
        self.Y = self.X.reshape(self.yshp)
    def backward(self):
        self.dX = self.dY.reshape(self.X.shape)


    @property
    def dims(self):
        return self.__dims
    @dims.setter
    def dims(self, value):
        value = np.array(value)
        if np.sum(value == -1) > 1:
            raise ValueError("There is one -1 at most.")
        self.__dims = value
