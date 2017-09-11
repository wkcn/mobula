from .Layer import *

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

        self.W = None
        self.b = None
    def reshape(self):
        # (NCHW)
        N,C,H,W = self.X.shape
        self.NH = (H + self.pad_h * 2 - self.kernel_h) // self.stride + 1
        self.NW = (W + self.pad_w * 2 - self.kernel_w) // self.stride + 1
        self.Y = np.zeros((N, self.dim_out, self.NH, self.NW))
        # Convolution Core
        if self.W is None:
            self.NHW = self.NH * self.NW
            self.W = Xavier((self.dim_out, C, self.kernel_w * self.kernel_h))
            self.b = np.zeros((self.dim_out, self.NHW))
            self.PH = H + self.pad_h * 2
            self.PW = W + self.pad_w * 2
            # X_col index
            self.I = im2col(np.arange(self.PH * self.PW).reshape((self.PH, self.PW)), (self.kernel_h, self.kernel_w), self.stride).ravel()

        B = im2col(np.pad(np.arange(H * W).reshape((H, W)), ((self.pad_h, self.pad_h), (self.pad_w, self.pad_w)), "constant", constant_values = -1), (self.kernel_h, self.kernel_w), self.stride).ravel()
        bb = (B != -1)
        self.Bb = np.tile(bb, N*C) # boolean
        tb = B[bb]
        #[0, HW, 2HW, 3HW...] * len(tb)
        self.Bi = np.repeat(np.arange(N * C) * (H*W), len(tb)) + np.tile(tb, N * C) 

    def get_col(self, X):
        N,C,H,W = self.X.shape
        if self.pad_h != 0 or self.pad_w != 0:
            pad = np.pad(X, ((0,0),(0,0),(self.pad_h,self.pad_h),(self.pad_w,self.pad_w)), "constant")
            return pad.reshape((N,C,self.PH * self.PW))[:,:,self.I].reshape((N,C,self.kernel_h * self.kernel_w,self.NHW))
        return X.reshape((N,C,self.PH * self.PW))[:,:,self.I].reshape((N,C,self.kernel_h * self.kernel_w,self.NHW))
    def forward(self):
        self.X_col = self.get_col(self.X) # (N, C, kernel_h * kernel_w, NH * NW) 
        self.Y = (np.tensordot(self.X_col, self.W, axes = ([1,2],[1,2])).swapaxes(1,2) + self.b).reshape(self.Y.shape)
    def backward(self):
        N,C,H,W = self.X.shape
        dY = self.dY.reshape((N,self.dim_out,self.NHW))
        # Update dW
        # dY: (N,D,W')
        # X_col: (N,C,H',W')
        # dW: (N,D,C,H')
        # W: (D, C, H')
        self.dW = np.tensordot(dY, self.X_col, axes = ([0, 2],[0, 3])) / N
        # Update db
        self.db =  np.mean(dY, 0) 
        # Update dX
        dX_col = np.tensordot(dY, self.W, axes = ([1],[0])).transpose((0, 2, 3, 1)) # (N, C, H', W')
        self.dX = npg.aggregate(self.Bi, dX_col.ravel()[self.Bb], size = self.X.size).reshape(self.X.shape)
    @property
    def params(self):
        return [self.W, self.b]
    @property
    def grads(self):
        return [self.dW, self.db]
