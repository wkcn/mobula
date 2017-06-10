from Layer import *

class FC(Layer):
    def __init__(self, model, *args, **kwargs):
        Layer.__init__(self, model, *args, **kwargs)
        self.W = np.zeros((0,0))
        self.b = np.arange(0)
        self.XN = 0
        self.C = 0
        self.dim_out = kwargs["dim_out"]
    def __str__(self):
        return "It is a Full Conneted Layer"
    def reshape(self):
        # NCHW
        self.XN = self.X.shape[0]
        self.C = self.X.size // self.XN 
        self.W = Xavier((self.dim_out ,self.C))
        self.b = Xavier((self.dim_out, 1))
        self.Y = np.zeros((self.XN, self.dim_out))
    def forward(self):
        # Y = W * X + b
        self.Y = np.dot(self.X.reshape((self.XN, self.C)), self.W.T) + self.b.T
    def reshape2(self):
        self.dX = np.zeros(self.X.shape)
        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)
    def backward(self):
        self.dX = np.dot(self.dY, self.W)
        self.dW = np.dot(self.dY.T, self.X.reshape((self.XN, self.C))) / self.XN
        self.db = np.mean(self.dY, 0).T.reshape(self.db.shape)
    def update(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db
