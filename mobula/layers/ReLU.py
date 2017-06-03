from Layer import *

class ReLU(Layer):
    def __init__(self, model, *args, **kwargs):
        Layer.__init__(self, model, *args, **kwargs)
    def __str__(self):
        return "It is a ReLU Layer"
    def reshape(self):
        self.Y = np.zeros(self.X.shape)
    def reshape2(self):
        self.dX = np.zeros(self.X.shape)
    def forward(self):
        self.Y = self.X.copy()
        self.Y[self.X < 0] = 0.0
    def backward(self):
        self.dX = self.dY.copy()
        self.dX[self.X < 0] = 0.0

if __name__ == "__main__":
    from Data import *
    X = np.matrix("-2,3;-1,5;-10,-6")
    data = Data(X, "data") 
    relu = ReLU(data, "relu")
    relu.forward()
    print relu.Y
    print X
    relu.dY = X
    relu.backward()
    print relu.dX
    print X
