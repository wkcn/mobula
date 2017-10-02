from .LossLayer import *

class SoftmaxWithLoss(LossLayer):
    def __init__(self, model, *args, **kwargs):
        LossLayer.__init__(self, model, *args, **kwargs)
        self.axis = kwargs.get("axis", 1) # N,C,H,W
    def reshape(self):
        self.Y = 0.0 
    def forward(self):
        e = np.exp(self.X - np.max(self.X, axis = self.axis, keepdims = True))
        s = np.sum(e, axis = self.axis, keepdims = True)
        self.softmax = e / s
        self.idx = get_idx_from_arg(self.softmax, self.label, self.axis)
        self.Y = -np.mean(np.log(self.softmax.ravel()[self.idx]))
    def backward(self):
        self.dX = self.softmax.copy()
        self.dX.ravel()[self.idx] -= 1.0
        self.dX *= self.dY
