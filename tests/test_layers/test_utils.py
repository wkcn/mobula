from mobula.layers.utils.Defines import * 
import numpy as np

def test_im2col():
    N, C, H, W = 3,4,5,6
    X = np.arange(N*C*H*W).reshape((N,C,H,W))
    def go_im2col(stride, KH, KW):
        print (stride, KH, KW)
        NH = (H - KH) // stride + 1
        NW = (W - KW) // stride + 1
        Y = np.zeros((N, C, KH * KW, NH * NW))
        T = np.zeros((N, C, KH * KW, NH * NW))
        for i in range(N):
            for c in range(C):
                for ch in range(NH):
                    for cw in range(NW):
                        sh = ch * stride
                        sw = cw * stride
                        Y[i, c, :, ch * NW + cw] = X[i, c, sh:sh+KH, sw:sw+KW].ravel()
                T[i, c] = im2col(X[i,c], (KH, KW), stride)
        #print (T[0,0], "====", Y[0,0])
        assert np.allclose(T, Y) 


    for stride in [1, 2, 3, 5]: 
        for KH in range(2, 5):
            for KW in range(2, 5):
                go_im2col(stride, KH, KW)

def test_idx_arg():
    # X = np.random.random((3,4,5,6))
    X = np.random.random((1,2,3,4))
    for i in range(4):
        idx = np.argmax(X, axis = i)
        ma = np.max(X, axis = i)
        y = get_val_from_idx(X, get_idx_from_arg(X, idx, axis = i)).reshape(ma.shape)
        assert np.allclose(ma, y)

def test_blocks():
    N,C,H,W = 3,4,9,9
    K = 3
    X = np.random.random((N,C,H,W))
    Y = np.random.random((N,C,H//K,W//K))
    for i in range(N):
        for c in range(C):
            for rr in range(H // K):
                for cc in range(W // K):
                    tr = rr * K
                    tc = cc * K
                    Y[i,c,rr,cc] = np.max(X[i,c,tr:tr+K,tc:tc+K])
    blocks = get_blocks(X, (K,K))
    assert np.allclose(np.max(blocks, (4,5)), Y)

