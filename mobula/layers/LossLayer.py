from .Layer import *

class LossLayer(Layer):
    def __init__(self, model, *args, **kwargs):
        Layer.__init__(self, model, *args, **kwargs)
        #self.label = kwargs.get("label") # (N, C)
        self.label_data = kwargs.get("label_data")
    @property
    def label(self):
        return self.label_data.label
    @property
    def loss(self):
        return 0.0
    @loss.setter
    def loss(self, value):
        raise ("You can't change loss value :-(")
