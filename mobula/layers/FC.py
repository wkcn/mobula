from .Layer import *

class FC(Layer):
    def __init__(self, model, *args, **kwargs):
        Layer.__init__(self, model, *args, **kwargs)
        self.W = None 
        self.b = None 
        self.XN = 0
        self.C = 0
        self.dim_out = kwargs["dim_out"]
    def reshape(self):
        # NCHW
        self.XN = self.X.shape[0]
        self.C = self.X.size // self.XN 
        self.Y = np.zeros((self.XN, self.dim_out))
        if self.W is None:
            self.W = Xavier((self.dim_out ,self.C))
            self.b = np.zeros((self.dim_out, 1))

    def forward(self):
        # Y = W * X + b
        self.Y = np.dot(self.X.reshape((self.XN, self.C)), self.W.T) + self.b.T
    def backward(self):
        self.dX = np.dot(self.dY, self.W)
        self.dW = np.dot(self.dY.T, self.X.reshape((self.XN, self.C))) / self.XN
        self.db = np.mean(self.dY, 0).reshape(self.b.shape)
    @property
    def params(self):
        return [self.W, self.b]
    @property
    def grads(self):
        return [self.dW, self.db]
