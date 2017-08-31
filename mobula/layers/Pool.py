from .Layer import *

class Pool(Layer):
    MAX, AVG = range(2)
    def __init__(self, model, *args, **kwargs):
        Layer.__init__(self, model, *args, **kwargs)
        if "kernel" in kwargs:
            self.kernel_w = kwargs["kernel"]
            self.kernel_h = kwargs["kernel"]
        else:
            self.kernel_w = kwargs["kernel_w"]
            self.kernel_h = kwargs["kernel_h"]
        self.stride = kwargs.get("stride", 1)
        self.pool_id = Pool.MAX # MAX: 0, AVG: 1
        pool = kwargs["pool"]
        if type(pool) == str:
            pool = pool.lower()
            if pool == "avg" or pool == "mean":
                self.pool_id = Pool.AVG
            elif pool == "max":
                self.pool_id = Pool.MAX
        elif type(pool) == int:
            self.pool_id = pool 
        else:
            raise RuntimeError("Pool key Error: ", self.pool)
    def reshape(self):
        # (NCHW)
        N,C,H,W = self.X.shape
        self.NH = (H - self.kernel_h) // self.stride + 1
        self.NW = (W - self.kernel_w) // self.stride + 1
        self.Y = np.zeros((N, C, self.NH, self.NW))
        self.NHW = self.NH * self.NW
        self.I = im2col(np.arange(H * W).reshape((H, W)), (self.kernel_h, self.kernel_w), self.stride).ravel()
        self.KHW = self.kernel_h * self.kernel_w
        B = im2col(np.arange(H * W).reshape((H, W)), (self.kernel_h, self.kernel_w), self.stride).ravel()
        self.Bi = np.repeat(np.arange(N * C) * (H*W), len(B)) + np.tile(B, N * C) 
    def get_col(self, X):
        N,C,H,W = self.X.shape
        return X.reshape(N,C,H * W)[:,:,self.I].reshape((N,C,self.kernel_h * self.kernel_w,self.NHW))
    def forward(self):
        self.X_col = self.get_col(self.X) # (N, C, kernel_h * kernel_w, NH * NW) 
        # self.X_col -> (N, C, 1, NH * NW)
        if self.pool_id == Pool.AVG:
            self.Y = np.mean(self.X_col, 2).reshape(self.Y.shape)
        else:
            maxI = np.argmax(self.X_col, 2).ravel()
            self.idx = get_idx_from_arg(self.X_col, maxI, axis = 2) # index of (N, C, KHW, NHW) 
            self.Y = get_val_from_idx(self.X_col, self.idx).reshape(self.Y.shape)
    def backward(self):
        N,C,H,W = self.X.shape
        if self.pool_id == Pool.AVG:
            # AVG
            dX_col = np.tile(self.dY.reshape((N, C, 1, self.NHW)), (1, 1, self.KHW, 1)) / self.KHW 
            self.dX = npg.aggregate(self.Bi, dX_col.ravel(), size = self.X.size).reshape(self.X.shape) 
        else:
            # MAX
            self.dX = npg.aggregate(self.Bi.ravel()[self.idx], self.dY.ravel(), size = self.X.size).reshape(self.X.shape)
