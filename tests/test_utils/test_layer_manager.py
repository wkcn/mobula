import mobula as M
import mobula.layers as L
import numpy as np

def test_layer_manager():
    X = np.arange(16).reshape((8, 2))

    # Notice: The time of importing LayerManager is only Once when nosetests 

    s = "/test_layer_manager/"
    with M.name_scope("test_layer_manager"):
        data0 = L.Data(X)
        data1 = L.Data(X)
        data2 = L.Data(X)
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

def test_layer_manager_in_function():
    s = "/test_layer_manager/"
    with M.name_scope("test_layer_manager"):
        X = np.arange(16).reshape((8, 2))
        data0 = L.Data(X)

        def hello():
            data1 = L.Data(X)
            with M.name_scope("mobula"):
                data2 = L.Data(X)
            print ("in", data1.name)
            assert data1.name == s + "Data_1"
            assert data2.name == s + "mobula/Data"

        hello()
        data3 = L.Data(X)

        assert data0.name == s + "Data"
        assert data3.name == s + "Data_1"

def test_get_layer():
    X = np.arange(16).reshape((8, 2))
    data0 = L.Data(X)
    data0_ref = M.get_layer(data0.name)
    assert data0_ref is data0
    data0_ref = M.get_layer("/Data")
    assert data0_ref is data0
    data0_ref = M.get_layer("Data")
    assert data0_ref is data0

def test_name():
    L.Add([L.ReLU(None), L.ReLU(None)])
    L.ReLU(L.ReLU(None))
    print (L.Add(None).name, L.ReLU(None).name)
    assert L.Add(None).name == "/Add"
    assert L.ReLU(None).name == "/ReLU"
    w = L.ReLU(L.ReLU(None)) # the first ReLU op is ReLU, and the second ReLU op is ReLU_1.
    assert L.ReLU(None).name == "/ReLU_2"
    del w
    assert L.ReLU(None).name == "/ReLU"
    net = M.Net()
    X = np.arange(16).reshape((2,2,2,2))
    print (L.Data().name)
    d1 = L.Data(X)
    d2 = L.Data(X)
    net.set_loss(d1 + d2)
    del d1, d2, net
    print (L.Data().name, L.Add(None).name)
    assert L.Data().name == "/Data"
    assert L.Add(None).name == "/Add"

def test_scope_name():
    assert M.get_scope_name() == "/"
    with M.name_scope("a"): 
        assert M.get_scope_name() == "/a/"
        with M.name_scope("b"): 
            assert M.get_scope_name() == "/a/b/"
            with M.name_scope("c"): 
                assert M.get_scope_name() == "/a/b/c/"
            assert M.get_scope_name() == "/a/b/"
        assert M.get_scope_name() == "/a/"
    assert M.get_scope_name() == "/"

def test_get_layers():
    x = np.ones((1,2,1,2))
    data0 = L.Data(x)
    relu0 = L.ReLU(x)
    with M.name_scope("mobula"):
        data1 = L.Data(x)
    scope = M.get_scope()
    root_layers = scope.get_layers()
    assert data0 in root_layers
    assert relu0 in root_layers
    assert data1 in root_layers
    assert len(root_layers) == 3
    mobula_layers = M.get_scope("mobula").get_layers()
    assert data0 not in mobula_layers
    assert relu0 not in mobula_layers
    assert data1 in mobula_layers
    assert len(mobula_layers) == 1
