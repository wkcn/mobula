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
