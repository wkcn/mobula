#coding=utf-8
import copy
import time
import pickle
import numpy as np
from .layers.MultiInput import *
from .layers.MultiOutput import *
from . import solvers

try:
    import queue
except:
    import Queue as queue
class Net(object):
    TRAIN, TEST = range(2)
    PHASE = TRAIN
    def __init__(self):
        self.loss = [] 
        self.topo = []
        self.set_solver(solvers.SGD)
    def set_loss(self, lossLayers):
        if type(lossLayers) != list:
            lossLayers = [lossLayers]
        self.loss = copy.copy(lossLayers)

        def R(mdr):
            if isinstance(mdr, YLayer):
                return mdr.model
            return mdr

        # Count
        q = queue.Queue()
        for l in lossLayers: 
            q.put(l)
        vis = set()
        cs = dict() # in degree
        while not q.empty():
            l = q.get()
            if l in vis:
                continue
            vis.add(l)
            # if layer l has input
            # Data.model is None
            if l.model is not None:
                if isinstance(l.model, MultiInput):
                    for mdr in l.model:
                        md = R(mdr)
                        cs[md] = cs.get(md, 0) + 1
                        q.put(md)
                else:
                    md = R(l.model)
                    cs[md] = cs.get(md, 0) + 1
                    q.put(md)
        # Find
        q = queue.Queue()
        for l in lossLayers: 
            q.put(l)
        st = []
        while not q.empty():
            l = q.get()
            st.append(l)
            if l.model is not None:
                if isinstance(l.model, MultiInput):
                    for mdr in l.model:
                        md = R(mdr)
                        cs[md] -= 1
                        if cs[md] == 0:
                            q.put(md)
                else:
                    md = R(l.model)
                    cs[md] -= 1
                    if cs[md] == 0:
                        q.put(md)
        self.topo = st[::-1]
        t = time.time()
        for l in self.topo:
            l.forward_time = 0.0
            l.backward_time = 0.0
            self.forward_times = 0
            self.backward_times = 0

        self.reshape()
        self.init_solver()

    def set_solver(self, solver):
        self.solver = solver()
        self.init_solver()
    def reshape(self):
        for l in self.topo:
            l.reshape()
    def init_solver(self):
        if self.solver is not None:
            for l in self.topo:
                self.solver.init(l)
    def forward(self):
        self.forward_times += 1
        for l in self.topo:
            t = time.time()
            l.forward()
            l.forward_time += time.time() - t 
    def backward(self):
        self.backward_times += 1

        self.solver.update_lr(self.backward_times)

        for l in self.topo[::-1]:
            t = time.time()
            num_next_layers = len(l.next_layers)
            if num_next_layers > 0:
                if num_next_layers == 1:
                    l.dY = l.next_layers[0].dX
                else:
                    l.dY = np.zeros(l.dY.shape) 
                    for e in l.next_layers:
                        l.dY += e.dX
            l.backward()

            self.solver.update(l)

            l.backward_time += time.time() - t 
    def time(self):
        if self.forward_times == 0 or self.backward_times == 0:
            return
        print ("name\t|forward_time\t|backward_time\t|forward_mean\t|backward_mean\t|forward_times: %d, backward_times: %d" % (self.forward_times, self.backward_times))
        for l in self.topo:
            print ("%s\t|%f\t|%f\t|%f\t|%f" % (l.layer_name, l.forward_time, l.backward_time, l.forward_time / self.forward_times, l.backward_time / self.backward_times))
    def save(self, filename):
        '''
        Save the learning parameters of network by layer_name
        '''
        print ("Saving the parameters of the network to %s:" % filename)
        data = []
        for l in self.topo: 
            params = l.params
            if len(params):
                data.append((l.layer_name, [p.tostring() for p in params]))
                print (" - %s" % l.layer_name)
        fout = open(filename, "wb")
        pickle.dump(data, fout, protocol = 2)
        fout.close()
        print ("Saving Finished :-)")
    def load(self, filename):
        '''
        Load the learning parameters of network by layer_name
        '''
        print ("Loading the parameters of the network from %s:" % filename)
        fin = open(filename, "rb")
        data = pickle.load(fin)
        fin.close()
        mapping = {}
        for i in range(len(data)):
            mapping[data[i][0]] = i
        for l in self.topo:
            if l.layer_name in mapping:
                lp = data[mapping[l.layer_name]][1]
                for j in range(len(l.params)):
                    p = l.params[j]
                    l.params[j][...] = np.fromstring(lp[j], dtype = p.dtype).reshape(p.shape) 
                print (" - %s" % l.layer_name)
        print ("Loading Finished :-)")
    @property
    def lr(self):
        return self.solver.lr
    @lr.setter
    def lr(self, value):
        self.solver.base_lr = value

# For compatibility
Net.setLoss = Net.set_loss
