from Layer import *

class Conv(Layer):
    def __init__(self, model, *args, **kwargs):
        Layer.__init__(self, model, *args, **kwargs)
        self.dim_out = kwargs.get("dim_out")
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
    def pad(self, X): 
        if self.pad_w == 0 and self.pad_h == 0:
            return self.X
        T = np.zeros((self.X.shape[0], self.X.shape[1], self.X.shape[2] + self.pad_h * 2, self.X.shape[3] + self.pad_w * 2))
        for n in range(self.X.shape[0]):
            for c in range(self.X.shape[1]):
                T[n,c,:,:] = np.pad(X[n,c,:,:], ((self.pad_h, self.pad_h), (self.pad_w, self.pad_w)), "constant")
        return T
    def forward(self):
        self.padX = self.pad(self.X)
        print self.padX
