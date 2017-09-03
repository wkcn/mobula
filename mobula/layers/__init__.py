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

# Add Methods
def get_func(op):
    def get_layer(lhs, rhs):
        l = op([lhs, rhs], "op")
        l.reshape()
        return l
    return lambda lhs, rhs : get_layer(lhs, rhs) 

Layer.__add__ = get_func(Add) 
YLayer.__add__ = get_func(Add) 
