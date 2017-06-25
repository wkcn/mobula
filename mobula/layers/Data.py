from .Layer import *

class Data(Layer):
    def __init__(self, model, layer_name = "", *args, **kwargs):
        self.layer_name = layer_name
        self.model = None
        self._Y = model
        self.dim_out = model.shape[1]
        self.next_layers = []
        self.lr = 0.0
        self.batch_size = kwargs.get("batch_size")
        self.batch_i = 0
        self._label = kwargs.get("label")
        self.N = len(self._Y)
        self.rounds = 0 
    def __str__(self):
        return "It is a Data Layer"
    def reset_index(self):
        self.batch_i = 0
    def set_data(self, Y):
        self._Y = Y
        self.rounds = 0
        self.N = len(self._Y)
    def set_label(self, label):
        self._label = label
    def reshape(self):
        if self.batch_size:
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
    @property
    def label(self):
        if self.batch_size is None:
            return self._label
        return self._label[self.index]
