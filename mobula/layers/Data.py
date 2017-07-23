from .Layer import *

class Data(Layer):
    def __init__(self, model, layer_name = "", *args, **kwargs):
        self.layer_name = layer_name
        self.model = None

        self.set_data(model)

        self.next_layers = []
        self.lr = 0.0
        self.batch_size = kwargs.get("batch_size")
        self.batch_i = 0

        self.set_label(kwargs.get("label"))
    def __str__(self):
        return "It is a Data Layer"
    def reset_index(self):
        self.batch_i = 0
    def set_data(self, Y):
        shp = list(Y.shape)
        shp.extend([1 for _ in range(4 - len(shp))])
        self._Y = Y.reshape(shp)
        self.rounds = 0
        self.N = len(self._Y)
    def set_label(self, label):
        if label is None:
            self._label = None
            return
        if len(label.shape) == 1:
            self._label = label.reshape((label.size, 1))
        else:
            self._label = label
    def reshape(self):
        if self.batch_size is not None:
            self.index = [0] * self.batch_size
    def forward(self):
        if self.batch_size is None:
            return
        i = 0
        self.index = []
        while i < self.batch_size:
            bi = self.batch_i
            be = bi + self.batch_size - i
            if be >= self.N: 
                be = self.N
                self.batch_i = 0
                self.rounds += 1
            else:
                self.batch_i = be 
            self.index.extend([k for k in range(bi, be)])
            i += (be - bi)
    @property
    def Y(self):
        if self.batch_size is None:
            return self._Y
        return self._Y[self.index]
    @Y.setter
    def Y(self, value):
        self.set_data(value)
    @property
    def label(self):
        if self.batch_size is None:
            return self._label
        return self._label[self.index]
    @label.setter
    def label(self, value):
        self.set_label(value)
