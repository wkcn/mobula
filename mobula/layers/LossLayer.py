from .Layer import *
from .utils.VModel import *

class LossLayer(Layer):
    def __init__(self, model, *args, **kwargs):
        Layer.__init__(self, model, *args, **kwargs)
        if "label" in kwargs:
            label = kwargs["label"]
            if isinstance(label, np.ndarray): 
                self.__label = VModel()
                self.__label.Y = label
            else:
                self.__label = label 
        self.dY = np.array(1.0)
        self.Y = 0.0
    @property
    def loss(self):
        return np.mean(self.Y)
    @property
    def label(self):
        return self.__label.Y
    @property
    def dY(self):
        return self.__dY.ravel()
    @dY.setter
    def dY(self, value):
        self.__dY = value
    @property
    def Y(self):
        return self.__Y.reshape((1,1,1,1))
    @Y.setter
    def Y(self, value):
        self.__Y = np.array(value)
