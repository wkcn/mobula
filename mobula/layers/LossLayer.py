from .Layer import *

class LossLayer(Layer):
    def __init__(self, model, *args, **kwargs):
        Layer.__init__(self, model, *args, **kwargs)
        self.__label = kwargs.get("label") 
    @property
    def label(self):
        return self.__label.Y
    @property
    def loss(self):
        return 0.0
    @loss.setter
    def loss(self, value):
        raise ("You can't change loss value :-(")
