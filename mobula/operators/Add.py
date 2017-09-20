from .Layer import *

class Add(Layer):
    def __init__(self, models, *args, **kwargs):
        Layer.__init__(self, models, *args, **kwargs)
    def reshape(self):
        self.Y = np.zeros(self.X[0].shape)
    def forward(self):
        self.Y = np.sum(self.X, 0) 
    def backward(self):
        self.dX = [self.dY for _ in range(len(self.X))]

class AddConstant(Layer):
    def __init__(self, model, *args, **kwargs):
        self.check_inputs(model, 1)
        Layer.__init__(self, model, *args, **kwargs)
        self.constant = kwargs["constant"]
    def reshape(self):
        self.Y = np.zeros(self.X.shape)
    def forward(self):
        self.Y = self.X + self.constant
    def backward(self):
        self.dX = self.dY 

Add.OP_L = AddConstant
Add.OP_R = AddConstant
