from Layer import *

class FC(Layer):
    def __init__(self, model, *args, **kwargs):
        Layer.__init__(self, model, *args, **kwargs)
        self.W = np.zeros((0,0))
        self.b = np.arange(0)
        self.XN = 0
        self.C = 0
    def __str__(self):
        return "It is a Full Conneted Layer"
    def reshape(self):
        # NCHW
        self.XN, XC, XH, XW = self.X.shape
        self.C = XC * XH * XW
        self.W = np.zeros((self.dim_out ,self.C))
    def forward(self):
        # Y = W * X + b
        self.X.shape = (self.XN, self.C)
        self.Y = np.dot(self.X, self.W.T) + self.b.T

class FCGradient(LayerGradient):
    def __init__(self, model):
        self.model = model
        self.dX = np.zeros(model.X.shape)
        self.dY = np.zeros(model.Y.shape)
        self.dW = np.zeros(model.W.shape)
        self.db = np.zeros(model.b.shape)
    def backward(self):
        self.dX = np.dot(self.dY, self.model.W)
        n = self.dY.shape[0]
        self.dW = np.zeros(model.W.shape)
        for i in range(n):
            self.dW += np.dot(self.dY[i, :].T, self.model.X[i, :])
        self.dW /= n
        self.db = np.mean(self.dY, 0).T
    def update(self, lr):
        self.model.W -= self.dW * lr
        self.model.b -= self.db * lr
