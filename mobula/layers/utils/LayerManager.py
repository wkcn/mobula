import weakref
from .Saver import *

class Scope(dict):
    def __init__(self, name = "", parent = None):
        dict.__init__(self)
        self.global_name = ("" if parent is None else parent.global_name) + "%s/" % name 

    def get_layers(self):
        layers = []
        for name, scope in self.items():
            if name[-1] != '/':
                # layer
                layers.append(scope())
            else:
                # scope
                layers.extend(scope.get_layers())
        return layers 

class LayerManagerClass(object):
    def __init__(self):
        self.root = Scope()
        self.cur_scope = self.root
        self.stack_scope = []

    def new_layer(self, name, obj, auto_rename = False):
        scope = self.get_scope(name)
        local_name = self.get_local_name(name, scope, auto_rename) 
        global_name = scope.global_name + local_name
        scope[local_name] = weakref.ref(obj)
        return global_name

    def del_layer(self, name):
        scope = self.get_scope(name)
        local_name = name.split('/')[-1]
        del scope[local_name]
        if len(scope) == 0:
            if scope != self.root:
                self.del_scope(name)

    def get_layer(self, name):
        scope = self.get_scope(name)
        local_name = name.split('/')[-1]
        return scope[local_name]()

    def get_scope(self, name):
        is_root = (name[0] == '/')
        if is_root:
            scope = self.root
            sp = name.split('/')[1:-1]
        else:
            scope = self.cur_scope
            sp = name.split('/')[:-1]
        for sn in sp:
            snp = sn + '/'
            if snp not in scope:
                scope[snp] = Scope(sn, scope)
            scope = scope[snp]
        return scope

    def del_scope(self, name):
        '''
            global_name | len(sp)
            ------------|--------
            /           | 2
            /a/         | 3
            /a/b/       | 4
            /a/b/c/     | 5
        '''
        st = []
        sp = name.split('/')[1:-1]
        scope = self.root
        for sn in sp:
            snp = sn + '/'
            scope = scope[snp]
            st.append((snp, scope)) # local_name, scope
        for i in range(len(st) - 1, 0, -1): # exclude 0
            snp, scope = st[i]
            if len(scope) == 0:
                del st[i - 1][1][snp]
            else:
                break

    def get_local_name(self, name, scope, auto_rename):
        if name in scope:
            if auto_rename:
                i = 1 
                while True:
                    new_name = "%s_%d" % (name, i)
                    if new_name not in scope:
                        break
                    i += 1
                name = new_name
            else:
                raise NameError("Duplicate Layer Name %s" % global_name)
        return name

    def enter_scope(self, name):
        if name[-1] != '/':
            name += '/'
        # add global_name
        name = self.cur_scope.global_name + name

        self.stack_scope.append(self.cur_scope)
        self.cur_scope = self.get_scope(name)

    def exit_scope(self):
        self.cur_scope = self.stack_scope.pop()

    def get_scope_name(self):
        return self.cur_scope.global_name

    def save(self, filename, scope_name):
        scope = self.get_scope(scope_name)
        save_layers(filename, scope.get_layers())
        print ("Saving Scope %s to %s Finished :-)" % (scope_name, filename))

    def load(self, filename, scope_name):
        scope = self.get_scope(scope_name)
        load_layers(filename, scope.get_layers())
        print ("Loading Scope %s from %s Finished :-)" % (scope_name, filename))

LayerManager = LayerManagerClass()
