#coding=utf-8
from .Defines import *
from .layers.utils.MultiInput import *
from .layers.utils.MultiOutput import *
from . import solvers
import functools
import signal
import weakref

try:
    import queue
except:
    import Queue as queue

class Net(object):
    def __init__(self):
        self.topo = []
        self.layers = dict()
        self.set_solver(solvers.SGD())
        self.phase = TRAIN
        # signal.signal(signal.SIGINT, self.signal_handler) 
    def set_loss(self, lossLayers):
        if type(lossLayers) != list:
            lossLayers = [lossLayers]

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
            # l.model may be Layer or MultiInput
            if l.model is not None:
                for md in l.model.input_models():
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
                for md in l.model.input_models():
                    cs[md] -= 1
                    if cs[md] == 0:
                        q.put(md)
        self.topo = st[::-1]

        self.layers = dict()
        for l in self.topo:
            self.layers[l.name] = l
            l.forward_time = 0.0
            l.backward_time = 0.0
            self.forward_times = 0
            self.backward_times = 0
            # Avoid bidirection reference
            l.net = weakref.proxy(self)

        self.reshape()
        for l in lossLayers: 
            l.dY = np.ones(l.Y.shape)
        self.init_solver()

    def set_solver(self, solver):
        self.solver = solver
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
                    l.dY = np.zeros(l.Y.shape) 
                    for e in l.next_layers:
                        l.dY += e.dX
            # compute the gradient dX of layer l
            l.backward()
            # use the solver to update weights of layer l
            if l.lr > 0:
                self.solver.update(l)

            l.backward_time += time.time() - t 
    def time(self):
        if self.forward_times == 0 or self.backward_times == 0:
            return
        print ("name\t|forward_time\t|backward_time\t|forward_mean\t|backward_mean\t|forward_times: %d, backward_times: %d" % (self.forward_times, self.backward_times))
        for l in self.topo:
            print ("%s\t|%f\t|%f\t|%f\t|%f" % (l.name, l.forward_time, l.backward_time, l.forward_time / self.forward_times, l.backward_time / self.backward_times))
    def save(self, filename):
        '''
        Save the learning parameters of network by name
        '''
        print ("Saving the parameters of the network to %s:" % filename)
        data = []
        for l in self.topo: 
            params = l.params
            if len(params):
                data.append((l.name, [p.tostring() for p in params]))
                print (" - %s" % l.name)
        fout = open(filename, "wb")
        pickle.dump(data, fout, protocol = 2)
        fout.close()
        print ("Saving Finished :-)")
    def load(self, filename):
        '''
        Load the learning parameters of network by name
        '''
        print ("Loading the parameters of the network from %s:" % filename)
        fin = open(filename, "rb")
        data = pickle.load(fin)
        fin.close()
        mapping = {}
        for i in range(len(data)):
            mapping[data[i][0]] = i
        for l in self.topo:
            if l.name in mapping:
                lp = data[mapping[l.name]][1]
                for j in range(len(l.params)):
                    p = l.params[j]
                    l.params[j][...] = np.fromstring(lp[j], dtype = p.dtype).reshape(p.shape) 
                print (" - %s" % l.name)
        print ("Loading Finished :-)")
    def __getitem__(self, name):
        return self.layers.get(name)
    @property
    def lr(self):
        return self.solver.lr
    @lr.setter
    def lr(self, value):
        self.solver.base_lr = value
    def signal_handler(self, signal, frame):
        # TODO: Exit to Save
        print ("Exit")
        pass

# For compatibility
Net.setLoss = Net.set_loss
