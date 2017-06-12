from .Defines import * 
class Layer(object):
    def __init__(self, model, layer_name = "", *args, **kwargs):
        # NCHW
        # (batch_size, dim_in, H, W)
        # V
        # (batch_size, dim_out, H, W)
        self.model = model
        self.next_layers = []
        self.model.next_layers.append(self)
        self.layer_name = layer_name
        self.args = args
        self.kwargs = kwargs
        self.lr = kwargs.get("lr", 1.0)
        self.Y = np.zeros((0,0,0,0))
    def reshape(self):
        pass
    def reshape2(self):
        pass
    def forward(self):
        pass
    def backward(self):
        pass
    def update(self, lr):
        pass
    @property
    def shape(self):
        return self.Y.shape
    @shape.setter
    def shape(self, value):
        raise RuntimeError("Don't Change Layer.shape")
    @property
    def X(self):
        return self.model.Y
    @X.setter
    def X(self, value):
        raise RuntimeError("Don't Change Layer.X")
