import sys
import os

PATH = os.path.dirname(__file__)
sys.path.append(PATH)
sys.path.append(PATH + "/../../thirdparty")

# Data Layer
from .Data import *

# Layers
from .FC import *
from .Conv import *
from .ConvT import *
from .BatchNorm import *

# Layers without learning
from .Pool import *
from .Dropout import *
from .Reshape import *
from .Crop import *

# Activate Layer
from .Sigmoid import *
from .ReLU import *
from .PReLU import *
from .SELU import *
from .Tanh import *
from .Softmax import *

# Multi IO Layer
from .Concat import *
from .Slice import *
from .Eltwise import *

# Cost Layer
from .MSE import *
from .CrossEntropy import *
from .SigmoidCrossEntropy import *
from .SoftmaxWithLoss import *
from .L1Loss import *
from .SmoothL1Loss import *
from .ContrastiveLoss import *

# Evaluation Layer (No Backward)
from .Accuracy import *

# Operators
from .Add import *
from .Subtract import *
from .Multiply import *
from .Negative import *

# Test Layer
from .MergeTest import *
from .SplitTest import *

def get_op2(op):
    def get_layer(lhs, rhs):
        args = [lhs, rhs] # array
        l = op(args, "op")
        return l
    return get_layer

add = get_op2(Add)
subtract = get_op2(Subtract)
multiply = get_op2(Multiply)
for l in [Layer, YLayer]:
    l.__add__ = add 
    l.__radd__ = add 
    l.__mul__ = multiply
    l.__rmul__ = multiply
    l.__pos__ = lambda x : x
    l.__neg__ = lambda x : Negative(x)
    l.__sub__ = subtract 
