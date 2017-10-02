from .utils.Defines import * 
from .utils.MultiInput import *
from .utils.MultiOutput import *
from .utils.VModel import *
from .utils.NullNet import *
from .utils.LayerManager import *

def is_layer(l):
    return isinstance(l, Layer) or isinstance(l, YLayer)

def is_layer_lst(lst):
    for l in lst:
        if not is_layer(l):
            return False
    return True

class Layer(object):
    def __init__(self, model, *args, **kwargs):
        # NCHW
        # (batch_size, dim_in, H, W)
        # V
        # (batch_size, dim_out, H, W)

        model = self.base_init(model, *args, **kwargs)

        # bidirection
        if model is not None:
            if is_layer(model):
                # Single Input 
                self.model = model
                self.model.next_layers.append(WeakrefLayer(self))
            elif type(model) == list:
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
            elif type(model) == np.ndarray:
                # For test
                self.model = VModel()
                self.model.Y = model
            else:
                raise TypeError("model must be a Layer or a List") 
        else:
            # for test
            self.model = VModel() 

        self.net = NullNet 

    def base_init(self, model, *args, **kwargs):
        name = None
        if model is not None:
            # For more models parameters
            if type(model) != list and len(args) > 0:
                lst = [model]
                for a in args:
                    if is_layer(a) or type(a) == np.ndarray:
                        lst.append(a)
                    else:
                        break
                args = args[len(lst) - 1:]
                if len(lst) > 1:
                    model = lst

        if "name" in kwargs:
            name = kwargs["name"]
        elif len(args) > 0 and type(args[0]) == str:
            name = args[0]

        # Name
        auto_rename = False 
        if name is None:
            name = self.__class__.__name__
            auto_rename = True
        self.name = LayerManager.new_layer(name, self, auto_rename = auto_rename)

        self.next_layers = []
        self.lr = kwargs.get("lr", 1.0)
        self.args = args
        self.kwargs = kwargs
        return model

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
    def input_models_with_index(self):
        yield (get_layer_parent(self), 0)
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
        return hasattr(self, "Y")
    def __str__(self):
        if not hasattr(self, "Y"):
            num_output = 1 # For not Data Layer
        elif self.Y is None:
            num_output = 0 # For Data Layer
        elif type(self.Y) != list:
            num_output = 1
        else:
            num_output = len(self.Y)

        input_names = ["%s:%d" % (l.name, i) for l,i in self.model.input_models_with_index()] if self.model is not None else []
        if len(input_names) > 1:
            input_info = 'input: [' + ','.join(input_names) + ']'
        elif len(input_names) == 1:
            input_info = "input: %s" % input_names[0]
        else:
            input_datas = None
            if type(self.model) == VModel:
                input_datas = self.model.Y # For VModel
            elif hasattr(self, "datas"):
                input_datas = self.datas # For Data Layer

            if input_datas is not None:
                # For Data Layer
                if type(input_datas) == list:
                    input_info = "input: %s" % str([d.shape for d in input_datas])
                else:
                    input_info = "input: %s" % str(input_datas.shape)
            else:
                input_info = "input: None"
        return "<%s '%s' %s num_output: (%d)>" % (self.__class__.__name__, self.name, input_info, num_output)
    __repr__ = __str__
    def __del__(self):
        LayerManager.del_layer(self.name)
    def check_inputs(self, models, num_inputs):
        if type(models) != list:
            return
        assert len(models) == num_inputs, "the number of inputs of %s must be %d" % (self.__class__.__name__, num_inputs)
