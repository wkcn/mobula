from .Layer import *

class Subtract(Layer):
    def __init__(self, models, *args, **kwargs):
        self.check_inputs(models, 2)
        Layer.__init__(self, models, *args, **kwargs)
    def reshape(self):
        self.Y = np.zeros(self.X[0].shape)
    def forward(self):
        self.Y = self.X[0] - self.X[1] 
    def backward(self):
        self.dX = [self.dY, -self.dY]

class SubtractConstantL(Layer):
    def __init__(self, model, *args, **kwargs):
        self.check_inputs(model, 1)
        Layer.__init__(self, model, *args, **kwargs)
        self.constant = kwargs["constant"]
    def reshape(self):
        self.Y = np.zeros(self.X.shape)
    def forward(self):
        self.Y = self.X - self.constant
    def backward(self):
        self.dX = self.dY 

class SubtractConstantR(Layer):
    def __init__(self, model, *args, **kwargs):
        self.check_inputs(model, 1)
        Layer.__init__(self, model, *args, **kwargs)
        self.constant = kwargs["constant"]
    def reshape(self):
        self.Y = np.zeros(self.X.shape)
    def forward(self):
        self.Y = self.constant - self.X
    def backward(self):
        self.dX = -self.dY 

Subtract.OP_L = SubtractConstantL
Subtract.OP_R = SubtractConstantR
