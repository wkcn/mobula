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
from .SmoothL1Loss import *
from .ContrastiveLoss import *

# Evaluation Layer (No Backward)
from .Accuracy import *

# Operators
from .Add import *

# Test Layer
from .MergeTest import *
from .SplitTest import *

# [TODO...]Add Methods
# Notice: It will merge the same operator.
# Example:
# l = L.Add([a,b], "add")
# w = l + c, namely: L.Add([a,b,c], "op") rather than L.Add([l, c], "op")
# l is discarded.

def get_func(op):
    def get_layer(lhs, rhs):
        def layer_expand(hs):
            if type(hs) == op:
                return hs.model.model
            return [hs]

        args = layer_expand(lhs) + layer_expand(rhs)
        l = op(args, "op")
        l.reshape()
        return l
    return get_layer

for op in [Add]:
    for l in [Layer, YLayer]:
        f = get_func(op)
        l.__add__ = f 
        l.__radd__ = f 
