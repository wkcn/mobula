from .Layer import *

class LossLayer(Layer):
    def __init__(self, model, *args, **kwargs):
        Layer.__init__(self, model, *args, **kwargs)
        self.label = kwargs.get("label") # (N, C)
        self.label_data = kwargs.get("label_data")
    def update_label(self):
        if self.label_data:
            self.label = self.label_data.label
    @property
    def loss(self):
        return 0.0
    @loss.setter
    def loss(self, value):
        raise ("You can't change loss value :-(")
