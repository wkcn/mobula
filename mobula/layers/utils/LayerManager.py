import weakref

class LayerManager(object):
    LAYERS = dict()
    NAME_SCOPE = ""
    @staticmethod
    def new_layer(name, obj, auto_rename = False):
        name = LayerManager.NAME_SCOPE + name
        if name in LayerManager.LAYERS:
            if auto_rename:
                i = 1 
                while True:
                    new_name = "%s_%d" % (name, i)
                    if new_name not in LayerManager.LAYERS:
                        break
                    i += 1
                name = new_name
            else:
                raise NameError("Duplicate Layer Name")
        LayerManager.LAYERS[name] = weakref.ref(obj)
        return name
    @staticmethod
    def del_layer(name):
        del LayerManager.LAYERS[name]
    @staticmethod
    def get_layer(name):
        return LayerManager.LAYERS[name]() # get the instance
