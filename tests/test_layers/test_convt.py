import mobula.layers as L
import numpy as np

def go_convt(stride, pad):
    print ("test ConvT: ", stride, pad)
    X = np.random.random((2, 4, 4, 4)) * 100
    N, D, NH, NW = X.shape
    K = 3
    C = 1
    FW = np.random.random((D, C, K * K)) * 10
    F = FW.reshape((D, C, K, K))

    data = L.Data(X)
    convT = L.ConvT(data, kernel = K, pad = pad, stride = stride, dim_out = C)

    pad_h = pad_w = pad
    kernel_h = kernel_w = K

    OH = (NH - 1) * stride + kernel_h - pad_h * 2 
    OW = (NW - 1) * stride + kernel_w - pad_w * 2 

    data.reshape()
    convT.reshape()

    convT.W = FW
    convT.b = np.random.random(convT.b.shape) * 10

    # Conv:  (OH, OW) -> (NH, NW)
    # ConvT: (NH. NW) -> (OH, OW)
    influence = [[[None for _ in range(kernel_h * kernel_w)] for _ in range(OW)] for _ in range(OH)]
    for h in range(NH):
        for w in range(NW):
            for fh in range(kernel_h):
                for fw in range(kernel_w):
                    ph = h * stride + fh
                    pw = w * stride + fw
                    oh = ph - pad_h
                    ow = pw - pad_w
                    if oh >= 0 and ow >= 0 and oh < OH and ow < OW:
                        influence[oh][ow][fh * kernel_w + fw] = (h, w)

    ty = np.zeros((N, C, OH, OW))
    dW = np.zeros(convT.W.shape)
    dX = np.zeros(convT.X.shape)
    dY = np.random.random(convT.Y.shape) * 100
    # F = FW.reshape((D, C, K, K))
    # N, D, NH, NW = X.shape
    for i in range(N):
        for c in range(C):
            for oh in range(OH):
                for ow in range(OW):
                    il = influence[oh][ow]
                    for t, pos in enumerate(il):
                        if pos is not None:
                            h,w = pos
                            for d in range(D):
                                ty[i, c, oh, ow] += X[i, d, h, w] * FW[d, c].ravel()[t]
                                dW[d, c].ravel()[t] += dY[i, c, oh, ow] * X[i, d, h, w]
                                dX[i, d, h, w] += dY[i, c, oh, ow] * FW[d, c].ravel()[t]

    ty += convT.b

    db = np.mean(dY, 0)
    dW /= N

    convT.forward()
    assert np.allclose(convT.Y, ty)

    # test backward
    # db, dw, dx
    convT.dY = dY
    convT.backward()

    assert np.allclose(convT.db, db)
    assert np.allclose(convT.dW, dW)
    assert np.allclose(convT.dX, dX)


def test_conv():
    go_convt(1, 0)
    go_convt(2, 0)
    go_convt(3, 0)
    go_convt(1, 1)
    go_convt(2, 1)
    go_convt(3, 1)
