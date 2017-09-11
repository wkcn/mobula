from .LossLayer import *

class L1Loss(LossLayer):
    def __init__(self, model, *args, **kwargs):
        LossLayer.__init__(self, model, *args, **kwargs)
    def reshape(self):
        self.Y = 0.
    def forward(self):
        self.Y = np.mean(np.abs(self.X))
    def backward(self):
        self.dX = np.zeros(self.X.shape)
        self.dX[self.X < 0] = -1
        self.dX[self.X > 0] = 1
        self.dX *= self.dY
