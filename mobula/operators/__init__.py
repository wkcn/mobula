from operator import add
from functools import reduce

from ..layers import *
from .Add import *
from .Subtract import *
from .Multiply import *
from .Positive import *
from .Negative import *
from .MatMul import *
Dot = MatMul # alias of Mulmat

from .Power import *
from .Exp import *
from .Log import *
from .Abs import *
from .ReduceMean import *
from .ReduceMax import *
from .ReduceMin import *
from .Compare import *

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
            elif is_constant(lhs):
                return op.OP_R(rhs, constant = lhs)

        args = [lhs, rhs] # array
        l = op(args)
        return l
    return get_layer

# operators
add = get_op2(Add)
subtract = get_op2(Subtract)
subtract_r = get_op2(Subtract, right = True)
multiply = get_op2(Multiply)
matmul = dot = get_op2(MatMul)
matmul_r = dot_r = get_op2(MatMul, right = True)
positive = lambda x : Positive(x) if is_constant(x) else x
negative = lambda x : Negative(x) 
reduce_mean = ReduceMean
reduce_max = ReduceMax
reduce_min = ReduceMin
equal = get_op2(Equal)
not_equal = get_op2(NotEqual)
greater_equal = get_op2(GreaterEqual)
greater = get_op2(Greater)
less_equal = get_op2(LessEqual)
less = get_op2(Less)
exp = lambda x : Exp(x)
log = lambda x : Log(x)
abs = lambda x : Abs(x)

for l in [Layer, YLayer]:
    l.__add__ = add 
    l.__radd__ = add 
    l.__mul__ = multiply
    l.__rmul__ = multiply
    l.__pos__ = positive 
    l.__neg__ = negative 
    l.__sub__ = subtract 
    l.__rsub__ = subtract_r
    l.__ge__ = greater_equal
    l.__gt__ = greater
    l.__le__ = less_equal
    l.__lt__ = less
