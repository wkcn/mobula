import pickle
import numpy as np

def save_layers(filename, layers, info = False):
    data = []
    for l in layers: 
        params = l.params
        if len(params) > 0:
            data.append((l.name, [p.tostring() for p in params]))
            if info:
                print (" - %s" % l.name)
    fout = open(filename, "wb")
    pickle.dump(data, fout, protocol = 2)
    fout.close()

def load_layers(filename, layers, info = False):
    fin = open(filename, "rb")
    data = pickle.load(fin)
    fin.close()
    mapping = {}
    for i in range(len(data)):
        mapping[data[i][0]] = i
    for l in layers:
        if l.name in mapping:
            lp = data[mapping[l.name]][1]
            for j in range(len(l.params)):
                p = l.params[j]
                l.params[j][...] = np.fromstring(lp[j], dtype = p.dtype).reshape(p.shape) 
            if info:
                print (" - %s" % l.name)

