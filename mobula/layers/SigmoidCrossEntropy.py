from .LossLayer import *

class SigmoidCrossEntropy(LossLayer):
    def __init__(self, model, *args, **kwargs):
        LossLayer.__init__(self, model, *args, **kwargs)
    def reshape(self):
        self.Y = 0.0 
    def forward(self):
        e = np.clip(1.0 / (1.0 + np.exp(-self.X)), 1e-16, 1.0 - 1e-16)
        self.Y = (np.mean(- np.multiply(self.label, np.log(e)) - \
               np.multiply(1.0 - self.label, np.log(1.0 - e))))
    def backward(self):
        self.dX = (np.clip(1.0 / (1.0 + np.exp(-self.X)), 1e-16, 1.0 - 1e-16) - self.label) * self.dY
