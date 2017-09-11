from .LossLayer import *

class ContrastiveLoss(LossLayer):
    def __init__(self, models, *args, **kwargs):
        # models = [X1, X2, sim]
        LossLayer.__init__(self, models, *args, **kwargs)
        self.margin = kwargs.get("margin", 1.0)
    def forward(self):
        self.sim = (self.X[2] == 1).ravel()
        n = self.sim.shape[0]
        self.diff = self.X[0] - self.X[1]
        dist_sq = np.sum(np.square(self.diff.reshape((n, -1))), 1)
        self.dist = np.sqrt(dist_sq).reshape((n, 1))
        df = (self.margin - self.dist).ravel()
        self.bdf = ((~self.sim) & (df > 0))
        self.Y = (np.sum(dist_sq[self.sim]) + np.sum(np.square(df[self.bdf]))) / (2.0 * n)
    def backward(self):
        n = self.sim.shape[0]
        dX = np.zeros(self.X[0].shape)
        dX[self.sim] = self.diff[self.sim] / n
        dX[self.bdf] = (1.0 / n - self.margin / n / self.dist[self.bdf]) * self.diff[self.bdf]
        self.dX[0] = self.dX[1] = dX * self.dY
