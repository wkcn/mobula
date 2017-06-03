from Layer import *

class Data(Layer):
    def __init__(self, model, layer_name = "", *args, **kwargs):
        self.layer_name = layer_name
        self.model = None
        self.Y = model
        self.dim_out = model.shape[1]
        self.next_layers = []
        self.lr = 0.0
    def __str__(self):
        return "It is a Data Layer"
