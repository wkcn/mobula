import numpy as np
from .Layer import *

class BatchNorm(Layer):
    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.momentum = kwargs.get('momentum', 0.9)
        self.eps = kwargs.get('eps', 1e-5)
        self.use_global_stats = kwargs.get('use_global_stats', False)
        self.axis = kwargs.get('axis', 1)
    def reshape(self):
        assert 0 <= self.axis < self.X.ndim
        self.cshape = [1] * self.X.ndim
        self.cshape[self.axis] = self.X.shape[self.axis]
        self.valid_axes = tuple([i for i in range(self.X.ndim) if i != self.axis])

        self.Y = np.zeros(self.X.shape)
        self.W = np.ones(self.cshape)
        self.b = np.zeros(self.cshape)
        self.moving_mean = np.zeros(self.cshape)
        self.moving_var = np.ones(self.cshape)
    def forward(self):
        if self.is_training() and not self.use_global_stats:
            self.batch_mean = np.mean(self.X, self.valid_axes, keepdims=True)
            self.batch_var = np.mean(np.square(self.X - self.batch_mean), self.valid_axes, keepdims=True)
            self.moving_mean = self.moving_mean * self.momentum + self.batch_mean * (1 - self.momentum)
            self.moving_var = self.moving_var * self.momentum + self.batch_var * (1 - self.momentum)
        else:
            self.batch_mean = self.moving_mean
            self.batch_var = self.moving_var
        self.nd = 1.0 / np.sqrt(self.batch_var + self.eps)
        self.nx = (self.X - self.batch_mean) * self.nd
        self.Y = np.multiply(self.nx, self.W) + self.b
    def backward(self):
        dnx = np.multiply(self.dY, self.W)
        xsm = self.X - self.batch_mean
        m = self.X.shape[0]
        dvar = np.sum(np.multiply(dnx, xsm), self.valid_axes, keepdims=True) * (-0.5) * np.power(self.nd, 3.0)
        dmean = -self.nd * np.sum(dnx, self.valid_axes, keepdims=True) - dvar * np.mean(xsm, self.valid_axes