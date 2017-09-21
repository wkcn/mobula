from .Layer import *
import operator
import functools

class Compare(Layer):
    def __init__(self, models, *args, **kwargs):
        self.check_inputs(models, 2)
        Layer.__init__(self, models, *args, **kwargs)
        self.op = kwargs["op"]
    def reshape(self):
        self.Y = np.zeros(self.X[0].shape)
    def forward(self):
        self.Y = self.op(self.X[0], self.X[1]) 
    def backward(self):
        self.dX = [np.zeros(self.X[0].shape), np.zeros(self.X[1].shape)]

class CompareConstantL(Layer):
    def __init__(self, model, *args, **kwargs):
        self.check_inputs(model, 1)
        Layer.__init__(self, model, *args, **kwargs)
        self.constant = kwargs["constant"]
        self.op = kwargs["op"]
    def reshape(self):
        self.Y = np.zeros(self.X.shape)
    def forward(self):
        self.Y = self.op(self.X, self.constant)
    def backward(self):
        self.dX = np.zeros(self.X.shape)

class CompareConstantR(Layer):
    def __init__(self, model, *args, **kwargs):
        self.check_inputs(model, 1)
        Layer.__init__(self, model, *args, **kwargs)
        self.constant = kwargs["constant"]
        self.op = kwargs["op"]
    def reshape(self):
        self.Y = np.zeros(self.X.shape)
    def forward(self):
        self.Y = self.op(self.constant, self.X)
    def backward(self):
        self.dX = np.zeros(self.X.shape)

class CompareTemplate(object):
    def __init__(self, op):
        self.op = op
    def __call__(self, args):
        return Compare(args, op = self.op)
    @property
    def OP_L(self):
        return functools.partial(CompareConstantL, op = self.op)
    @property
    def OP_R(self):
        return functools.partial(CompareConstantR, op = self.op)

def get_op(op):
    return CompareTemplate(op) 

Equal = get_op(operator.eq)
NotEqual = get_op(operator.ne)
GreaterEqual = get_op(operator.ge)
Greater = get_op(operator.gt)
LessEqual = get_op(operator.le)
Less = get_op(operator.lt)
