from .Layer import *

class BatchNorm(Layer):
    def __init__(self, model, *args, **kwargs):
        Layer.__init__(self, model, *args, **kwargs)
        self.eps = kwargs.get("eps", 1e-5)
    def reshape(self):
        self.Y = np.zeros(self.X.shape)
        self.W = np.ones(self.X.shape[1:]) 
        self.b = np.zeros(self.X.shape[1:]) 
        self.n = 0
        # Current Mean
        self.cmean = np.zeros(self.X.shape[1:])
        self.cvar = np.ones(self.X.shape[1:])
        # Global Mean
        self.gmean = np.zeros(self.X.shape[1:])
        self.gvar = np.ones(self.X.shape[1:])
    def forward(self):
        if self.is_training():
            self.n += 1
            a = (self.n - 1) * 1.0 / self.n
            # The mean and var of this batch
            self.cmean = np.mean(self.X, 0)
            self.cvar = np.var(self.X, 0)
            # The estimated maen and var
            self.gmean = a * self.gmean + self.cmean / self.n
            self.gvar = a * self.gvar + self.cvar / self.n
        else:
            self.cmean = self.gmean
            m = self.X.shape[0]
            self.cvar = m * 1.0 / (m - 1) * self.gvar
        # Normalize
        self.nd = 1.0 / np.sqrt(self.cvar + self.eps)
        self.nx = (self.X - self.cmean) * self.nd 
        # Scale and Shift
        self.Y = np.multiply(self.nx, self.W) + self.b
    def backward(self):
        # Compute self.dX, self.dW, self.db
        dnx = np.multiply(self.dY, self.W)
        xsm = self.X - self.cmean
        m = self.X.shape[0]
        dvar = np.sum(np.multiply(dnx,xsm),0) * (-0.5) * np.power(self.nd, 3.0)
        dmean = -self.nd * np.sum(dnx, 0) - dvar * np.mean(xsm, 0) * 2.0 
        self.dX = dnx * self.nd + dvar * xsm * (2.0 / m) + dmean * (1.0 / m)
        self.dW = np.sum(self.dY * self.nx, 0)
        self.db = np.sum(self.dY, 0)
    @property
    def params(self):
        return [self.W, self.b, self.gmean, self.gvar]
    @property
    def grads(self):
        return [self.dW, self.db]
