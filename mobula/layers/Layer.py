from .utils.Defines import * 
from .utils.MultiInput import *
from .utils.MultiOutput import *
from .utils.VModel import *
from .utils.NullNet import *
from .utils.LayerManager import *

class Layer(object):
    def __init__(self, model, name = None, *args, **kwargs):
        # NCHW
        # (batch_size, dim_in, H, W)
        # V
        # (batch_size, dim_out, H, W)

        def is_layer(l):
            return isinstance(l, Layer) or isinstance(l, YLayer)
        def is_layer_lst(lst):
            for l in lst:
                if not is_layer(l):
                    return False
            return True

        # bidirectiop
        self.next_layers = []
        if model is not None:
            if is_layer(model):
                if type(model.Y) == list and len(model.YLayers) == 1:
                    # For a output in MultiOutput Layer
                    self.model = model.YLayers[0]
                else:
                    self.model = model
                self.model.next_layers.append(self)
            elif isinstance(model, list):
                if is_layer_lst(model):
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
                    # for test
                    self.model = VModel() 
                    self.model.Y = model

            else:
                raise TypeError("model must be a Layer or a List") 
        else:
            # for test
            self.model = VModel() 

        if name is None:
            name = self.__class__.__name__

        self.name = LayerManager.new_layer(name, self, auto_rename = True)

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
        if num > 1:
            self.YLayers = [YLayer(self, i) for i in range(num)]
            self.Y = [None] * num
            self.dY = MultiDY(self) 
        else:
            self.YLayers = None
            self.Y = None
            self.dY = None
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
    def input_models(self):
        yield get_layer_parent(self)
    def forward_all(self):
        if self.model is not None:
            for md in self.model.input_models():
                md.forward_all()
        self.forward()
    def reshape_all(self):
        if self.model is not None:
            for md in self.model.input_models():
                md.reshape_all()
        self.reshape()
    def eval(self):
        self.forward_all()
        return self.Y
    def __del__(self):
        LayerManager.del_layer(self.name)
