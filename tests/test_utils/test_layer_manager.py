import mobula as M
import mobula.layers as L
import numpy as np

def test_layer_manager():
    X = np.arange(16).reshape((8, 2))

    data0 = L.Data(X)
    data1 = L.Data(X)
    data2 = L.Data(X)
    assert data0.name == "Data"
    assert data1.name == "Data_1"
    assert data2.name == "Data_2"

    relu0 = L.ReLU(data0)
    assert relu0.name == "ReLU"
    relu1 = L.ReLU(data1)
    assert relu1.name == "ReLU_1"

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

    assert data3.name == "wkcn/Data"
    assert data4.name == "wkcn/Data_1"
    assert data5.name == "Data_3"

    assert relu2.name == "wkcn/ReLU"
    assert relu3.name == "wkcn/ReLU_1"
    assert relu4.name == "wkcn/mobula/ReLU"
    assert relu5.name == "wkcn/ReLU_2"
    assert relu6.name == "ReLU_2"
