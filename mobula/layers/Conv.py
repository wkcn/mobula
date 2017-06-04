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
        self.NH = (self.X.shape[2] + self.pad_h * 2 - self.kernel_h) // self.stride
        self.NW = (self.X.shape[3] + self.pad_w * 2 - self.kernel_w) // self.stride
        self.Y = np.zeros((self.X.shape[0], self.dim_out, self.NH, self.NW))
        # Convolution Core
        # self.W = Xavier((self.dim_out, self.X.shape[1], self.kernel_h, self.kernel_w)) 
        self.W = Xavier((self.dim_out, self.X.shape[1] * self.kernel_w * self.kernel_w))
    def get_col(self, X):
        return np.stack([np.hstack(\
                [im2col(\
                np.pad(X[n,c,:,:], ((self.pad_h, self.pad_h), (self.pad_w, self.pad_w)), "constant"), \
                (self.kernel_h, self.kernel_w), self.stride) \
                for c in range(self.X.shape[1])]).T \
                for n in range(self.X.shape[0])])
    def forward(self):
        self.X_col = self.get_col(self.X)
        self.Y = np.dot(self.W, self.X_col).swapaxes(0, 1)
