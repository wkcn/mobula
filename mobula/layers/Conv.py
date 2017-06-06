from Layer import *

class Conv(Layer):
    def __init__(self, model, *args, **kwargs):
        Layer.__init__(self, model, *args, **kwargs)
        self.dim_out = kwargs["dim_out"]
        if "pad" in kwargs:
            self.pad_w = kwargs["pad"]
            self.pad_h = kwargs["pad"]
        else:
            self.pad_w = kwargs.get("pad_w", 0)
            self.pad_h = kwargs.get("pad_h", 0)
        if "kernel" in kwargs:
            self.kernel_w = kwargs["kernel"]
            self.kernel_h = kwargs["kernel"]
        else:
            self.kernel_w = kwargs["kernel_w"]
            self.kernel_h = kwargs["kernel_h"]
        self.stride = kwargs.get("stride", 1)
    def __str__(self):
        return "It is a Convolution Layer"
    def reshape(self):
        # (NCHW)
        N,C,H,W = self.X.shape
        self.NH = (H + self.pad_h * 2 - self.kernel_h + 1) // self.stride
        self.NW = (W + self.pad_w * 2 - self.kernel_w + 1) // self.stride
        self.NHW = self.NH * self.NW
        self.Y = np.zeros((N, self.dim_out, self.NH, self.NW))
        # Convolution Core
        # self.W = Xavier((self.dim_out, self.X.shape[1], self.kernel_h, self.kernel_w)) 
        self.W = Xavier((self.dim_out, C * self.kernel_w * self.kernel_h))
        self.b = Xavier((self.dim_out, self.NHW))
        self.I = im2col(np.pad(np.arange(H * W).reshape((H, W)), ((self.pad_h, self.pad_h), (self.pad_w, self.pad_w)), "constant", constant_values = -1), (self.kernel_h, self.kernel_w), self.stride)
    def get_col(self, X):
        return np.stack([np.vstack(\
                [im2col(\
                np.pad(X[n,c,:,:], ((self.pad_h, self.pad_h), (self.pad_w, self.pad_w)), "constant"), \
                (self.kernel_h, self.kernel_w), self.stride) \
                for c in range(X.shape[1])]) \
                for n in range(X.shape[0])])
    def forward(self):
        self.X_col = self.get_col(self.X)
        self.Y = (np.dot(self.W, self.X_col).swapaxes(0, 1) + self.b).reshape((self.X.shape[0], self.dim_out, self.NH, self.NW))
    def backward(self):
        N,C,H,W = self.X.shape
        self.dY.resize((N, self.dim_out, self.NHW))
        self.dW = np.mean([np.dot(self.dY[i, :, :], self.X_col[i,:,:].T) for i in range(self.X.shape[0])], 0)
        self.db = np.mean(self.dY, 0)

        dX_col = np.dot(self.W.T, np.asarray(self.dY)).swapaxes(0, 1)

        self.dX = np.zeros(self.X.shape)
        HW = H * W
        khw = self.kernel_h * self.kernel_w
        si = self.I.size
        self.I.resize(si)
        for n in range(N):
            for c in range(C):
                f = dX_col[n,khw * c:khw * (c+1), :].reshape(si) 
                np.add.at(self.dX[n,c,:,:].reshape(HW), self.I[self.I != -1], f[self.I != -1])
    def update(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db
