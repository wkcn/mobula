from .LossLayer import *

class SmoothL1Loss(LossLayer):
    def __init__(self, model, *args, **kwargs):
        LossLayer.__init__(self, model, *args, **kwargs)
    def reshape(self):
        self.Y = 0.
    def forward(self):
        # TODO: optimize
        a = np.abs(self.X)
        bm = (a < 1)
        bo = ~bm
        z = np.zeros(self.X.shape) 
        z[bm] = np.square(a[bm]) * 0.5
        z[bo] = a[bo] - 0.5
        self.Y = np.mean(z)
    def backward(self):
        br = (self.X >= 1)
        bl = (self.X <= -1)
        bm = ~(br | bl) # |x| < 1
        self.dX = np.ones(self.X.shape)
        #self.dX[br] = 1
        self.dX[bl] = -1
        self.dX[bm] = self.X[bm]
        self.dX *= self.dY
