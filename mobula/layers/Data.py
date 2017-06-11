from Layer import *

class Data(Layer):
    def __init__(self, model, layer_name = "", *args, **kwargs):
        self.layer_name = layer_name
        self.model = None
        self.allY = model
        self.dim_out = model.shape[1]
        self.next_layers = []
        self.lr = 0.0
        self.batch_size = kwargs.get("batch_size")
        self.batch_i = 0
        self._label = kwargs.get("label")
        self.N = len(self.allY)
    def __str__(self):
        return "It is a Data Layer"
    def reshape(self):
        if self.batch_size:
            yshape = [self.allY.shape[i] for i in range(len(self.allY.shape))]
            if self.batch_size < yshape[0]:
                yshape[0] = self.batch_size
            self.Y = np.zeros(yshape)
            if self._label != None:
                lshape = [self._label.shape[i] for i in range(len(self._label.shape))]
                if self.batch_size < lshape[0]:
                    lshape[0] = self.batch_size
                self.label = np.zeros(lshape) 
        else:
            self.Y = self.allY 
            self.label = self._label
    def forward(self):
        if self.batch_size == None:
            return
        i = 0
        while i < self.batch_size:
            bi = self.batch_i
            be = bi + self.batch_size - i
            if be >= self.N: 
                be = self.N
                self.batch_i = 0
            else:
                self.batch_i = be 
            j = be - bi + i 
            self.Y[i:j,:,:,:] = self.allY[bi:be,:,:,:]
            self.label[i:j, :] = self._label[bi:be, :]
            i = j
