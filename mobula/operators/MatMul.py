from .Layer import *

class MatMul(Layer):
    def __init__(self, models, *args, **kwargs):
        Layer.__init__(self, models, *args, **kwargs)
    def reshape(self):
        self.Y = np.dot(self.X[0], self.X[1])
    def forward(self):
        self.Y = np.dot(self.X[0], self.X[1])
    def backward(self):
        self.dX = [np.dot(self.dY, self.X[1].T), np.dot(self.X[0].T, self.dY)]

# Y = X @ constant
class MatMulConstantL(Layer):
    def __init__(self, models, *args, **kwargs):
        Layer.__init__(self, models, *args, **kwargs)
        self.constant = kwargs["constant"]
    def reshape(self):
        self.Y = np.dot(self.X, self.constant)
    def forward(self):
        self.Y = np.dot(self.X, self.constant)
    def backward(self):
        self.dX = np.dot(self.dY, self.constant.T)

# Y = constant @ X
class MatMulConstantR(Layer):
    def __init__(self, models, *args, **kwargs):
        Layer.__init__(self, models, *args, **kwargs)
        self.constant = kwargs["constant"]
    def reshape(self):
        self.Y = np.dot(self.constant, self.X)
    def forward(self):
        self.Y = np.dot(self.constant, self.X)
    def backward(self):
        self.dX = np.dot(self.constant.T, self.dY)

MatMul.OP_L = MatMulConstantL
MatMul.OP_R = MatMulConstantR
