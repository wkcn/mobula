from .layers.utils.LayerManager import *

class name_scope(object):
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        LayerManager.NAME_SCOPE += "%s/" % self.name
    def __exit__(self, *dummy):
        LayerManager.NAME_SCOPE = LayerManager.NAME_SCOPE[:-len(self.name)-1]

def get_layer(name):
    return LayerManager.get_layer(name)

# operators
from . import operators as O
add = O.add
subtract = O.subtract
multiply = O.multiply 
matmul = O.matmul 
dot = O.dot
positive = O.positive
negative = O.negative
reduce_mean = O.reduce_mean
reduce_max = O.reduce_max
reduce_min = O.reduce_min
equal = O.equal
not_equla = O.not_equal
greater_equal = O.greater_equal
greater = O.greater
less_equal = O.less_equal
less = O.less
