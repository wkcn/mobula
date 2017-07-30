from .utils.Defines import * 
from .utils.MultiInput import *
from .utils.MultiOutput import *
from .utils.VModel import *
from .utils.NullNet import *

class Layer(object):
    def __init__(self, model, name = "", *args, **kwargs):
        # NCHW
        # (batch_size, dim_in, H, W)
        # V
        # (batch_size, dim_out, H, W)

        # bidirectiop
        self.next_layers = []
        if model is not None:
            if isinstance(model, Layer) or isinstance(model, YLayer):
                if type(model.Y) == list and len(model.YLayers) == 1:
                    # For a output in MultiOutput Layer
                    self.model = model.YLayers[0]
                else:
                    self.model = model
                self.model.next_layers.append(self)
            elif isinstance(model, list):
                # It's a Multi Input Layer

                for i in range(len(model)):
                    if type(model[i].Y) == list and len(model[i].YLayers) == 1:
                        # For a output in MultiOutput Layer
                        model[i] = model[i].YLayers[0]

                self.model = MultiInput(model)
                for i in range(len(model)):
                    model[i].next_layers.append(XLayer(self, i))
                self.dX = [None] * len(model)
            else:
                raise TypeError("model must be a Layer or a List") 
        else:
            # for test
            self.model = VModel() 

        self.name = name
        self.args = args
        self.kwargs = kwargs
        self.lr = kwargs.get("lr", 1.0)
        self.Y = np.zeros((0,0,0,0))
        self.net = NullNet 
    def reshape(self):
        pass
    def forward(self):
        pass
    def backward(self):
        pass
    def update(self, lr):
        params = self.params
        grads = self.grads
        for i in range(len(params)):
            params[i] -= grads[i] * (lr * self.lr)
    @property
    def params(self):
        return []
    @property
    def grads(self):
        return []
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
        #raise RuntimeError("Don't Change Layer.X")
        self.model.Y = value
    def set_output(self, num):
        self.YLayers = [YLayer(self, i) for i in range(num)]
        self.Y = [None] * num
        self.dY = MultiDY(self) 
    def __call__(self, value = None):
        if type(self.Y) == list:
            # For MultiOutput
            if value is None:
                if len(self.Y) == 1:
                    return self.YLayers[0]
                return self.YLayers
            return self.YLayers[value]
        # For Single Output
        return self 
    def is_training(self):
        return self.net.phase == TRAIN
