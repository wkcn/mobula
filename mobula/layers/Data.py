from Layer import *

class Data(Layer):
    def __init__(self, model, layer_name = "", *args, **kwargs):
        self.layer_name = layer_name
        self.model = None
        self.allY = model
        self.dim_out = model.shape[1]
        self.next_layers = []
        self.lr = 0.0
        self.batch_size = kwargs.get("batch_size")
        self.batch_g = 0
        self._labels = kwargs.get("labels")
        tmp = self.Y
        self.labels = self._labels[:self.batch_size, :]
    def __str__(self):
        return "It is a Data Layer"
    @property
    def Y(self):
        if self.batch_size == None:
            return self.allY
        be = self.batch_size * self.batch_g
        en = be + self.batch_size
        if en >= self.allY.shape[0]:
            en = self.allY.shape[0]
            self.batch_g = 0
        else:
            self.batch_g += 1
        ty = self.allY[be:en,:,:,:]
        self.labels = self._labels[be:en, :]
        return ty
    @Y.setter
    def Y(self, value):
        raise RuntimeError("Don't Change Data.Y")
