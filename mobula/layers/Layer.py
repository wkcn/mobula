from Defines import * 
class Layer:
    BATCH_SIZE = 16
    def __init__(self, model, layer_name, dim_out, *args, **kwargs):
        # NHWC
        # (batch_size, H, W, dim_in)
        # V
        # (batch_size, H, W, dim_out)
        self.model = model
        self.layer_name = layer_name
        self.dim_out = dim_out
        self.args = args
        self.kwargs = kwargs
        self.Y = np.zeros((0,0,0,0))
        self.gradient = None
    def reshape(self):
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
    @property
    def X(self):
        return self.model.Y
