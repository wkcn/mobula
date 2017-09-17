from .MultiOutput import *
import weakref

class MultiInputItem(object):
    def __init__(self, models):
        self.models = models
    def __getitem__(self, i):
        return self.models[i].Y
    def __setitem__(self, i, value):
        self.models[i].Y = value
    def __iter__(self):
        for md in self.models:
            yield md.Y
    def __len__(self):
        return len(self.models)

class MultiInput(object):
    def __init__(self, model):
        self.model = model
        self.items = MultiInputItem(model)
    @property
    def Y(self):
        return self.items
    def __getitem__(self, i):
        return self.model[i]
    def __len__(self):
        return len(self.model)
    def __iter__(self):
        for md in self.model:
            yield md
    def input_models(self):
        for md in self.model:
            yield get_layer_parent(md)

class XLayer(object):
    def __init__(self, model, i):
        # Avoid bidirection reference
        self.model = weakref.ref(model)
        self.i = i
    @property
    def dX(self):
        return self.model().dX[self.i]

class WeakrefLayer(object):
    def __init__(self, model):
        # Avoid bidirection reference
        self.model = weakref.ref(model)
    @property
    def dX(self):
        return self.model().dX
