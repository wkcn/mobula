from .Solver import *

class Momentum(Solver):
    def __init__(self, *args, **kwargs):
        self.momentum = kwargs.get("momentum", 0.5)
        Solver.__init__(self, *args, **kwargs)
    def update(self, l):
        params = l.params
        grads = l.grads
        mlr = self.lr * l.lr
        for i in range(len(grads)):
            dx = self.momentum * l.mt[i] - mlr * grads[i]
            params[i] += dx 
            l.mt[i] = dx
    def init(self, l):
        l.mt = [0.0 for _ in range(len(l.params))]
