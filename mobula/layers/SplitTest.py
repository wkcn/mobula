from .Layer import *

# It's a test layer for multi output
# y1, y2 = SplitTest(x)()
# Notice: There is one more pair of brackets

class SplitTest(Layer):
    def __init__(self, model, *args, **kwargs):
        Layer.__init__(self, model, *args, **kwargs)
        self.set_output(2)
    def reshape(self):
        self.Y = [self.X[0::2], self.X[1::2]]
        self.dX = np.zeros(self.X.shape)
    def forward(self):
        self.Y = [self.X[0::2], self.X[1::2]]
    def backward(self):
        self.dX[0::2] = self.dY[0]
        self.dX[1::2] = self.dY[1]
