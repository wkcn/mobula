from .Layer import *

'''
[Notice]
If len(data) is not batch_size times, the batches are:
    Ex: data = [1,2,3,4,5,6,7,8,9,10], batch_size = 3
        batch_1: [1,2,3]
        batch_2: [4,5,6]
        batch_3: [7,8,9]
        batch_4: [2,3,4] # Notice!
        batch_5: [5,6,7]
        batch_6: [8,9,10]

    It's designed for performance.
    If order is nessary, it's better to set batch_size the factor of len(data)
'''

class Data(Layer):
    INPUT_TYPE_ERROR = "Data.datas must be a List with ndarrays or an ndarray"
    def __init__(self, datas = None, name = None, *args, **kwargs):
        # the type of datas is list with ndarrays or ndarray 

        # Type Check
        if datas is not None:
            if isinstance(datas, list):
                for d in datas:
                    if not isinstance(d, np.ndarray):
                        raise TypeError(Data.INPUT_TYPE_ERROR)
            elif isinstance(datas, np.ndarray):
                pass
            else:
                raise TypeError(Data.INPUT_TYPE_ERROR)


        auto_rename = False 
        if name is None:
            name = self.__class__.__name__
            auto_rename = True
        self.name = LayerManager.new_layer(name, self, auto_rename = auto_rename)

        self.next_layers = []
        self.lr = 0.0

        self.model = None
        self.__batch_size = None
        self.datas = datas
        self.batch_size = kwargs.get("batch_size")
    def reshape(self):
        if self.__batch_size is None:
            self.Y = self.__datas

        if type(self.__datas) != list:
            self.Y = self.__datas[:self.__batch_size] 
        else:
            self.Y = [data[:self.__batch_size] for data in self.__datas]

    def forward(self):
        if self.__batch_size is None:
            return
        e = self.__batch_i + self.__batch_size 
        if e > self.n:
            self.__batch_i = self.n - self.__batch_i
            e = self.__batch_i + self.__batch_size

        if type(self.__datas) != list:
            self.Y = self.__datas[self.__batch_i:e]
        else:
            self.Y = [data[self.__batch_i:e] for data in self.__datas]

        self.__batch_i = e

    @property
    def batch_size(self):
        return self.__batch_size
    @batch_size.setter
    def batch_size(self, value):
        self.__batch_size = value
        self.__batch_i = 0
        if value is None:
            self.Y = self.__datas
        if self.__datas is not None:
            self.reshape()
    @property
    def datas(self):
        return self.__datas
    @datas.setter
    def datas(self, value):
        self.__datas = value
        if value is None:
            return
        self.__batch_i = 0
        if type(self.__datas) == list:
            self.n = len(self.__datas[0])
            self.set_output(len(self.__datas))
        else:
            self.n = len(self.__datas)
            self.set_output(1)
        self.reshape()

    # Alias
    X = datas
