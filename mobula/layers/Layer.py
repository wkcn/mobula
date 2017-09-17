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

        auto_rename = False 
        if name is None:
            name = self.__class__.__name__
            auto_rename = True
        self.name = LayerManager.new_layer(name, self, auto_rename = auto_rename)

        self.next_layers = []
        self.lr = kwargs.get("lr", 1.0)
        self.args = args
        self.kwargs = kwargs

        # bidirection
        if model is not None:
            if is_layer(model):
                # Single Input 
                self.model = model
                self.model.next_layers.append(WeakrefLayer(self))
            elif isinstance(model, list):
                if is_layer_lst(model):
                    # It's a Multi Input Layer
                    self.model = MultiInput(model)
                    for i, mdi in enumerate(model):
                        mdi.next_layers.append(XLayer(self, i))
                    self.dX = [None] * len(model)
                else:
                    # for test
                    self.model = VModel() 
                    self.model.Y = model
            elif isinstance(model, np.ndarray):
                # For test
                self.model = VModel()
                self.model.Y = model
            else:
                raise TypeError("model must be a Layer or a List") 
        else:
            # for test
            self.model = VModel() 

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
            self.YLayers = [None] * num
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
                return self.get_YLayers()
            return self.get_YLayers(value)
        # For Single Output
        return self 
    def __iter__(self):
        if type(self.Y) == list:
            for y in self.get_YLayers():
                yield y
        else:
            yield self
    def get_YLayers(self, i = None):
        if i == None:
            lst = []
            for i in range(len(self.YLayers)):
                try:
                    assert self.YLayers[i] != None
                    l = self.YLayers[i]
                except:
                    l = YLayer(self, i)
                    self.YLayers[i] = weakref.proxy(l)
                lst.append(l)
            return lst


        try:
            assert self.YLayers[i] != None
            l = self.YLayers[i]
        except:
            l = YLayer(self, i)
            self.YLayers[i] = weakref.proxy(l)

        return l
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
    def eval(self, datas = None):
        need_to_reshape = False
        if datas is not None:
            need_to_reshape = self.whether_to_reshape(datas)
            for data, value in datas.items():
                data.X = value

        # check whether it has inited
        if need_to_reshape or not self.inited():
            self.reshape_all()
        self.forward_all()
        return self.Y
    def whether_to_reshape(self, datas):
        for data, value in datas.items():
            if type(data.X) != type(value):
                return True
            if type(data.X) != list:
                if data.X.shape != value.shape:
                    return True
            else:
                for a,b in zip(data.X, value):
                    if a.shape != b.shape:
                        return True
        return False
    def inited(self):
        try:
            type(self.Y)
        except:
            return False
        return True
    def __del__(self):
        LayerManager.del_layer(self.name)
