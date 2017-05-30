from Layer import *

class FC(Layer):
    def __init__(self, model, *args, **kwargs):
        Layer.__init__(self, model, *args, **kwargs)
        self.W = np.zeros((0,0))
        self.b = np.zeros((0,0))
        self.XN = 0
        self.C = 0
    def __str__(self):
        return "It is a Full Conneted Layer"
    def reshape(self):
        # NHWC
        self.XN, XH, XW, XC = self.X.shape
        self.C = XH * XW * XC
        # flatten self.X
        # Y = W * X + b
        # X.shape = (C, 1)
        # W.shape = (dim_out, C)
        # b.shape = (dim_out, 1)
        # Y.shape = (XN, C)
        self.W = np.zeros((self.dim_out, self.C))
        self.b = np.zeros((self.dim_out, 1)) 
        self.Y = np.zeros((self.XN, self.C))
    def forward(self):
        # Y = W * X + b
        self.Y = (self.X.reshape((self.XN, self.C)) * self.W.T + self.b.T)

class FCGradient:
    def __init__(self, model):
        self.model = model
        self.dX = np.zeros(model.X.shape)
        self.dY = np.zeros(model.Y.shape)
    def backward(self):
        pass
