from .layers.utils.LayerManager import *

class name_scope(object):
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        LayerManager.enter_scope(self.name)
    def __exit__(self, *dummy):
        LayerManager.exit_scope()

get_scope_name = LayerManager.get_scope_name
get_scope = LayerManager.get_scope
get_layer = LayerManager.get_layer
save_scope = LayerManager.save_scope 
load_scope = LayerManager.load_scope

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
