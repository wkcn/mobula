class YLayer:
    def __init__(self, model, i):
        self.model = model
        self.i = i
        self.dY = None
    @property
    def Y(self):
        return self.model.Y[self.i]
    @Y.setter
    def Y(self, value):
        self.model.Y[self.i] = value

class MultiDY:
    def __init__(self, model):
        self.model = model
    def __getitem__(self, i):
        return self.model.YLayers[i].dY
    def __setitem__(self, i, value):
        self.model.YLayers[i].dY = value
