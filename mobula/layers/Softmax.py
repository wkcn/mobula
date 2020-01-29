from .Layer import *

class Softmax(Layer):
    def __init__(self, model, *args, **kwargs):
        Layer.__init__(self, model, *args, **kwargs)
        self.axis = kwargs.get("axis", 1) # N,C,H,W
    def reshape(self):
        self.Y = np.zeros(self.X.shape)
    def forward(self):
        e = np.exp(self.X - np.max(self.X, axis = self.axis, keepdims = True))
        s = np.sum(e, axis = self.axis, keepdims = True)
        self.Y = e / s
    def backward(self):
        shape = self.X.shape
        # (N, C, P)
        N = int(np.prod(shape[:self.axis]))
        C = int(shape[self.axis])
        P = int(np.prod(shape[self.axis+1:]))

        '''
        dX = np.zeros((N, C, P))
        y = self.Y.reshape((N, C, P))
        dY = self.dY.reshape((N, C, P))
        for n in range(N):
            for p in range(P):
                # case: (n, p)
                for i in range(C):
                    for j in range(C):
                        if i == j:
                            dX[n, i, p] += (y[n, i, p] - y[n, i, p] ** 2) * dY[n, j, p]
                        else:
                            dX[n, i, p] += (-y[n, i, p] * y[n, j, p]) * dY[n, j, p]
        self.dX = dX.reshape(self.X.shape)
        '''

        y1 = self.Y.reshape((N, C, 1, P))
        y2 = self.Y.reshape((N, 1, C, P))

        s = (-y1) * y2
        # inds in (N, C, C, P)
        batch_inds = np.arange(0, (C*C*P)*N, C*C*P) # N
        # (n, 0, 0, k) -> (n, 1, 1, k) -> (n, 2, 2, k)
        diag_inds = np.arange(0, (P+C*P)*C, P+C*P) # C
        spatial_inds = np.arange(0, P) # P
        diag_inds = np.repeat(diag_inds, P)
        diag_inds += np.tile(spatial_inds, C)
        inds = np.repeat(batch_inds, C*P) + np.tile(diag_inds, N)
        # (N, C, C, P), self.Y: P->C->N
        s.ravel()[inds] += self.Y.ravel()
        # self.dY: (N, C, P)
        s *= self.dY.reshape(y2.shape)
        self.dX = s.sum(2).reshape(self.X.shape)
