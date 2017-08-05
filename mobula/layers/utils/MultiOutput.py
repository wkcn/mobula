class YLayer:
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
    def forward(self):
        self.model.forward()
    def backward(self):
        self.model.backward()
    def reshape(self):
        self.model.reshape()

class MultiDY:
    def __init__(self, model):
        self.model = model
    def __getitem__(self, i):
        return self.model.YLayers[i].dY
    def __setitem__(self, i, value):
        self.model.YLayers[i].dY = value
    def __iter__(self): 
        for dy in self.model.YLayers:
            yield dy
    def __len__(self):
        return len(self.model.YLayers)
