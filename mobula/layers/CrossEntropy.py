from .LossLayer import *

class CrossEntropy(LossLayer):
    def __init__(self, model, *args, **kwargs):
        LossLayer.__init__(self, model, *args, **kwargs)
    def __str__(self):
        return "It is a Mean Squared Error Layer"
    def reshape(self):
        self.Y = 0.
    def forward(self):
        self.update_label()
        self.Y = np.mean(- np.multiply(self.label, np.log(self.X)) - \
               np.multiply(1.0 - self.label, np.log(1.0 - self.X)))
    def reshape2(self):
        self.dX = np.zeros(self.X.shape) 
    def backward(self):
        self.dX = -self.label / self.X + (1.0 - self.label) / (1.0 - self.X)
    @property
    def loss(self):
        return self.Y
