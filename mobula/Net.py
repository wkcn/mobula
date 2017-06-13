#coding=utf-8
import copy
import time
try:
    import queue
except:
    import Queue as queue
class Net:
    def __init__(self):
        self.loss = [] 
        self.topo = []
        self.lr = 1.0
    def __str__(self):
        return "It is a network"
    def setLoss(self, lossLayers):
        if type(lossLayers) != list:
            lossLayers = [lossLayers]
        self.loss = copy.copy(lossLayers)
        # Count
        q = queue.Queue()
        for l in lossLayers: 
            q.put(l)
        vis = set()
        cs = dict()
        while not q.empty():
            l = q.get()
            if l in vis:
                continue
            vis.add(l)
            if l.model:
                cs[l.model] = cs.get(l.model, 0) + 1
                q.put(l.model)
        # Find
        q = queue.Queue()
        for l in lossLayers: 
            q.put(l)
        st = []
        while not q.empty():
            l = q.get()
            st.append(l)
            if l.model:
                cs[l.model] -= 1
                if cs[l.model] == 0:
                    q.put(l.model)
        self.topo = st[::-1]
        t = time.time()
        for l in self.topo:
            l.forward_time = 0.0
            l.backward_time = 0.0
            self.forward_times = 0
            self.backward_times = 0
    def reshape(self):
        for l in self.topo:
            l.reshape()
    def reshape2(self):
        for l in self.topo:
            l.reshape2()
    def forward(self):
        self.forward_times += 1
        for l in self.topo:
            t = time.time()
            l.forward()
            l.forward_time += time.time() - t 
    def backward(self):
        self.backward_times += 1
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
            l.update(self.lr * l.lr)
            l.backward_time += time.time() - t 
    def time(self):
        print ("name\t|forward_time\t|backward_time\t|forward_mean\t|backward_mean\t|forward_times: %d, backward_times: %d" % (self.forward_times, self.backward_times))
        for l in self.topo:
            print ("%s\t|%f\t|%f\t|%f\t|%f" % (l.layer_name, l.forward_time, l.backward_time, l.forward_time / self.forward_times, l.backward_time / self.backward_times))
