import mobula as M
import mobula.layers as L
import numpy as np

def test_layer_manager():
    X = np.arange(16).reshape((8, 2))

    '''
    Notice:
        The time of importing LayerManager is only Once when nosetests 
    '''
    s = "test_layer_manager/"
    with M.name_scope(s[:-1]):
        data0 = L.Data(X)
        data1 = L.Data(X)
        data2 = L.Data(X)
        print (data0.name)
        assert data0.name == s + "Data"
        assert data1.name == s + "Data_1"
        assert data2.name == s + "Data_2"

        relu0 = L.ReLU(data0)
        assert relu0.name == s + "ReLU"
        relu1 = L.ReLU(data1)
        assert relu1.name == s + "ReLU_1"

        with M.name_scope("wkcn"):
            data3 = L.Data(X) # wkcn/Data
            data4 = L.Data(X) # wkcn/Data_1
            relu2 = L.ReLU(data0) # wkcn/ReLU 
            relu3 = L.ReLU(data0) # wkcn/ReLU_1
            with M.name_scope("mobula"):
                relu4 = L.ReLU(data0) # wkcn/mobula/ReLU
            relu5 = L.ReLU(data0) # wkcn/ReLU_2

        data5 = L.Data(X) # Data_3
        relu6 = L.ReLU(data0) # ReLU_2

        assert data3.name == s + "wkcn/Data"
        assert data4.name == s + "wkcn/Data_1"
        assert data5.name == s + "Data_3"

        assert relu2.name == s + "wkcn/ReLU"
        assert relu3.name == s + "wkcn/ReLU_1"
        assert relu4.name == s + "wkcn/mobula/ReLU"
        assert relu5.name == s + "wkcn/ReLU_2"
        assert relu6.name == s + "ReLU_2"
