import weakref

class YLayer(object):
    def __init__(self, model, i):
        self.model = model
        self.i = i
        self.dY = None
        self.next_layers = []
    @property
    def Y(self):
        return self.model.Y[self.i]
    @Y.setter
    def Y(self, value):
        self.model.Y[self.i] = value
    @property
    def shape(self):
        return self.model.Y[self.i].shape
    def forward(self):
        self.model.forward()
    def backward(self):
        self.model.backward()
    def reshape(self):
        self.model.reshape()
    def input_models(self):
        yield self.model

class MultiDY(object):
    def __init__(self, model):
        self.model = weakref.proxy(model)
    def __getitem__(self, i):
        return self.model.get_YLayers(i).dY
    def __setitem__(self, i, value):
        self.model.get_YLayers(i).dY = value
    def __iter__(self): 
        for dy in self.model.get_YLayers():
            yield dy
    def __len__(self):
        return len(self.model.YLayers)

def get_layer_parent(md):
    if isinstance(md, YLayer):
        return md.model
    return md
