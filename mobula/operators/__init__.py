from operator import add
from functools import reduce

from ..layers import *
from .Add import *
from .Subtract import *
from .Multiply import *
from .Negative import *

from .Power import *
from .Exp import *
from .Log import *
from . ReduceMean import *
from . ReduceMax import *
from . ReduceMin import *

def is_constant(x):
    return not isinstance(x, Layer) and not isinstance(x, YLayer)

def get_op2(op, right = False):
    def get_layer(lhs, rhs):
        # Constant
        if right:
            return op.OP_R(lhs, constant = rhs)
        else:
            if is_constant(rhs):
                return op.OP_L(lhs, constant = rhs)

        args = [lhs, rhs] # array
        l = op(args)
        return l
    return get_layer

add = get_op2(Add)
subtract = get_op2(Subtract)
subtract_r = get_op2(Subtract, right = True)
multiply = get_op2(Multiply)
for l in [Layer, YLayer]:
    l.__add__ = add 
    l.__radd__ = add 
    l.__mul__ = multiply
    l.__rmul__ = multiply
    l.__pos__ = lambda x : x
    l.__neg__ = lambda x : Negative(x)
    l.__sub__ = subtract 
    l.__rsub__ = subtract_r
