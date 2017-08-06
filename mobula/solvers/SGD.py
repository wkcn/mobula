from .Solver import *

class SGD(Solver):
    def __init__(self, *args, **kwargs):
        Solver.__init__(self, *args, **kwargs)
    def update(self, l):
        params = l.params
        grads = l.grads
        mlr = self.lr * l.lr
        for i in range(len(grads)):
            params[i] -= grads[i] * mlr
