from .Layer import *

class Accuracy(Layer):
    def __init__(self, model, *args, **kwargs):
        Layer.__init__(self, model, *args, **kwargs)
        self.top_k = kwargs.get("top_k", 1)
        self.__label = kwargs["label"] 
    @property
    def label(self):
        return self.__label.Y
    def reshape(self):
        self.Y = 0.0
    def forward(self):
        w = np.argpartition(self.X, -self.top_k, axis = 1)[:, -self.top_k:]
        self.Y = np.mean((w == self.label).any(1))
