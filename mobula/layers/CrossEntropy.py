from .LossLayer import *

class CrossEntropy(LossLayer):
    def __init__(self, model, *args, **kwargs):
        LossLayer.__init__(self, model, *args, **kwargs)
    def reshape(self):
        self.Y = 0.0 
    def forward(self):
        self.Y = np.mean(- np.multiply(self.label, np.log(self.X)) - \
               np.multiply(1.0 - self.label, np.log(1.0 - self.X)))
    def backward(self):
        self.dX = (-self.label / self.X + (1.0 - self.label) / (1.0 - self.X)) * self.dY
