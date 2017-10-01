import mobula.layers as L
import numpy as np

def test_name():
    a = np.arange(10)
    b = np.zeros((3,5))
    test1 = [str(L.Data(a)), str(L.Data([a,b])), str(L.Data(None))]
    test1_truth = ["<Data '/Data' input: (10,) num_output: (1)>", "<Data '/Data' input: [(10,), (3, 5)] num_output: (2)>", "<Data '/Data' input: None num_output: (0)>"]
    print (test1)
    assert test1 == test1_truth

    test2 = [str(L.ReLU(L.Data(a))), str(L.ReLU(L.FC(L.Data(b), dim_out = 10)))]
    test2_truth = ["<ReLU '/ReLU' input: /Data num_output: (1)>", "<ReLU '/ReLU' input: /FC num_output: (1)>"]
    print (test2)
    assert test2 == test2_truth
