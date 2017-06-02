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
        self.Y = np.zeros((self.XN, self.dim_out))
    def forward(self):
        # Y = W * X + b
        self.X.shape = (self.XN, self.C)
        self.Y = np.dot(self.X, self.W.T) + self.b.T
        #print self.X.shape, self.W.T.shape, self.Y.shape
    def reshape2(self):
        self.dX = np.zeros(self.X.shape)
        self.dY = np.zeros(self.Y.shape)
        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)
    def backward(self):
        self.dX = np.dot(self.dY, self.W)
        n = self.dY.shape[0]
        self.dW = np.zeros(self.dW.shape)

        self.dY = np.asmatrix(self.dY)
        self.X = np.asmatrix(self.X)

        for i in range(n):
            self.dW += np.dot(self.dY[i, :].T, self.X[i, :])
        self.dW /= n
        self.db = np.mean(self.dY, 0).T
    def update(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db 
