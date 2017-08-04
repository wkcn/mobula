from .LossLayer import *

class Accuracy(LossLayer):
    def __init__(self, model, *args, **kwargs):
        LossLayer.__init__(self, model, *args, **kwargs)
        self.top_k = kwargs.get("top_k", 1)
    def shape(self):
        self.Y = 0.0
    def forward(self):
        w = np.argpartition(self.X, -self.top_k, axis = 1)[:, -self.top_k:]
        self.Y = np.mean((w == self.label).any(1))
    @property
    def loss(self):
        return 1.0 - self.Y
