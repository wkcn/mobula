from .layers.utils.LayerManager import *

class name_scope(object):
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        LayerManager.enter_scope(self.name)
    def __exit__(self, *dummy):
        LayerManager.exit_scope()

def get_layer(name):
    return LayerManager.get_layer(name)

def get_scope_name():
    return LayerManager.get_scope_name()

def get_scope(name = get_scope_name()):
    if name[-1] != '/':
        name += '/'
    return LayerManager.get_scope(name)

def get_layers(name = get_scope_name()):
    scope = get_scope(name)
    return scope.get_layers()

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
not_equal = O.not_equal
greater_equal = O.greater_equal
greater = O.greater
less_equal = O.less_equal
less = O.less

exp = O.exp
log = O.log
