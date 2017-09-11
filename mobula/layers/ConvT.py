from .Layer import *

class ConvT(Layer):
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
        N,D,NH,NW = self.X.shape
        self.NHW = NH * NW
        self.OH = (NH - 1) * self.stride + self.kernel_h - self.pad_h * 2 
        self.OW = (NW - 1) * self.stride + self.kernel_w - self.pad_w * 2 
        self.OHW = self.OH * self.OW
        self.Y = np.zeros((N, self.dim_out, self.OH, self.OW))
        if self.W is None:
            self.W = Xavier((D, self.dim_out, self.kernel_h * self.kernel_w))
            self.b = np.zeros((self.dim_out, self.OH, self.OW)) 
            self.PH = self.OH + self.pad_h * 2
            self.PW = self.OW + self.pad_w * 2
            self.I = im2col(np.arange(self.PH * self.PW).reshape((self.PH, self.PW)), (self.kernel_h, self.kernel_w), self.stride).ravel()

        B = im2col(np.pad(np.arange(self.OH * self.OW).reshape((self.OH, self.OW)), ((self.pad_h, self.pad_h), (self.pad_w, self.pad_w)), "constant", constant_values = -1), (self.kernel_h, self.kernel_w), self.stride).ravel()
        bb = (B != -1)
        self.Bb = np.tile(bb, N*self.dim_out) # boolean
        tb = B[bb]
        #[0, HW, 2HW, 3HW...] * len(tb)
        self.Bi = np.repeat(np.arange(N * self.dim_out) * (self.OH*self.OW), len(tb)) + np.tile(tb, N * self.dim_out) 

    def get_col(self, X):
        N,D,NH,NW = self.X.shape
        if self.pad_h != 0 or self.pad_w != 0:
            pad = np.pad(X, ((0,0),(0,0),(self.pad_h,self.pad_h),(self.pad_w,self.pad_w)), "constant")
            return pad.reshape((N,self.dim_out,self.PH * self.PW))[:,:,self.I].reshape((N,self.dim_out,self.kernel_h * self.kernel_w,self.NHW))
        return X.reshape((N,self.dim_out,self.PH * self.PW))[:,:,self.I].reshape((N,self.dim_out,self.kernel_h * self.kernel_w,self.NHW))

    def forward(self):
        '''
        ConvT   Conv
        dim_out C
        D       dim_out(D)
        -----------------
        X: (N, D, NH, NW)
        W: (D, self.dim_out, kh * kw)
        Y_col: (N, self.dim_out, kh * kw, NH * NW)
        '''
        N,D,NH,NW = self.X.shape
        Y_col = np.tensordot(self.X.reshape((N, D, self.NHW)), self.W, axes = ([1], [0])).transpose((0,2,3,1))
        self.Y = npg.aggregate(self.Bi, Y_col.ravel()[self.Bb], size = self.Y.size).reshape(self.Y.shape) + self.b 
    def backward(self):
        N,D,NH,NW = self.X.shape
        dY = self.dY.reshape((N,self.dim_out,self.OHW))
        self.db = np.mean(dY, 0).reshape((self.dim_out, self.OH, self.OW))
        dY_col = self.get_col(self.dY.reshape(self.Y.shape)) 
        X = self.X.reshape((N, D, self.NHW))
        self.dW = np.tensordot(X, dY_col, axes = ([0, 2], [0, 3])) / N
        self.dX = np.tensordot(dY_col, self.W, axes = ([1, 2], [1, 2])).swapaxes(1, 2).reshape(self.X.shape)
    @property
    def params(self):
        return [self.W, self.b]
    @property
    def grads(self):
        return [self.dW, self.db]
