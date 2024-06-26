from .LossLayer import *
import numpy as np

class CrossEntropy(LossLayer):
    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)

    def reshape(self):
        self.Y = 0.0 

    def forward(self):
        self.Y = np.mean(- np.multiply(self.label, np.log(self.X + 1e-15)) - \
               np.multiply(1.0 - self.label, np.log(1.0 - self.X + 1e-15)))

    def backward(self):
        self.dX = (-self.label / (self.X + 1e-15) + (1.0 - self.label) / (1.0 - self.X + 1e-15)) * self.dY
