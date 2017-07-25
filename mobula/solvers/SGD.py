from .Solver import *

class SGD(Solver):
    def __init__(self, *args, **kwargs):
        self.lr = kwargs.get("lr", 1.0)
        Solver.__init__(self, *args, **kwargs)
    def update(self, l):
        params = l.params
        grads = l.grads
        mlr = self.lr * l.lr
        for i in range(len(params)):
            params[i] -= grads[i] * mlr
