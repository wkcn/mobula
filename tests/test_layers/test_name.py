import mobula.layers as L
import numpy as np

def test_name():
    a = np.arange(10)
    b = np.zeros((3,5))
    c = np.arange(1, 11)
    test1 = [str(L.Data(a)), str(L.Data([a,b])), str(L.Data(None))]
    test1_truth = ["<Data '/Data' input: (10,) num_output: (1)>", "<Data '/Data' input: [(10,), (3, 5)] num_output: (2)>", "<Data '/Data' input: None num_output: (0)>"]
    print (test1)
    assert test1 == test1_truth, (test1, test1_truth)

    test2 = [str(L.ReLU(L.Data(a))), str(L.ReLU(L.FC(L.Data(b), dim_out = 10)))]
    test2_truth = ["<ReLU '/ReLU' input: /Data:0 num_output: (1)>", "<ReLU '/ReLU' input: /FC:0 num_output: (1)>"]
    print (test2)
    assert test2 == test2_truth, (test2, test2_truth)

    la, lc = L.Data([a,c])
    concat = L.Concat([la, lc], axis = 0)
    test3 = [str(concat)]
    test3_truth = ["<Concat '/Concat' input: [/Data:0,/Data:1] num_output: (1)>"]
    print (test3)
    assert test3 == test3_truth, (test3, test3_truth)

    l = L.ReLU(a)
    test4 = [str(l)]
    test4_truth = ["<ReLU '/ReLU' input: (10,) num_output: (1)>"]
    print (test4)
    assert test4 == test4_truth, (test4, test4_truth)
