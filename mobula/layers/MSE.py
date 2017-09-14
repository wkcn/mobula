from .LossLayer import *

class MSE(LossLayer):
    def __init__(self, model, *args, **kwargs):
        LossLayer.__init__(self, model, *args, **kwargs)
    def reshape(self):
        self.Y = 0.0 
    def forward(self):
        self.d = (self.X - self.label)
        self.Y = np.mean(np.square(self.d))
    def backward(self):
        self.dX = (2 * self.d) * self.dY
