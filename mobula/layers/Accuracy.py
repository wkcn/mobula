from .LossLayer import *

class Accuracy(LossLayer):
    def __init__(self, model, *args, **kwargs):
        LossLayer.__init__(self, model, *args, **kwargs)
        self.top_k = kwargs.get("top_k", 1)
    def reshape(self):
        self.Y = 0.0
    def forward(self):
        w = np.argpartition(self.X, -self.top_k, axis = 1)[:, -self.top_k:]
        self.Y = np.mean((w == self.label).any(1))
