import mobula.layers as L
import numpy as np

def go_conv(stride, pad):
    X = np.random.random((2, 4, 10, 10)) * 100
    W = np.random.random((2,4,9)) * 10
    F = W.reshape((2, 4, 3, 3))

    data = L.Data(X, "data")
    conv = L.Conv(data, "Conv", kernel = 3, pad = pad, stride = stride, dim_out = 2)

    pad_h = pad_w = pad
    kernel_h = kernel_w = 3

    XH = XW = 10
    NH = (XH + pad_h * 2 - kernel_h) // stride + 1
    NW = (XW + pad_w * 2 - kernel_w) // stride + 1

    data.reshape()
    conv.reshape()

    conv.W = W
    conv.b = np.random.random(conv.b.shape) * 10

    ty = np.zeros((2, 2, NH, NW))
    for i in range(2):
        x = X[i]
        x = np.pad(x, ((0,0),(pad,pad),(pad,pad)), "constant")
        for d in range(2):
            f = F[d] # 4, 3, 3
            for h in range(NH):
                for w in range(NW): 
                    th = h * stride
                    tw = w * stride
                    a = x[:, th:th+3, tw:tw+3]
                    e = np.sum(a * f)
                    ty[i, d, h, w] = e
        ty[i] += conv.b.reshape((2, NH, NW))
    conv.forward()
    print (np.max(np.abs(conv.Y - ty)), conv.Y.shape)
    assert np.allclose(conv.Y, ty)

    conv.dY = np.random.random(conv.Y.shape) * 50
    # test backward, dX, dW, db
    conv.backward()

    db = np.mean(conv.dY, 0)

    t = np.zeros((2, 4, 10, 10))
    dF = np.zeros((2, 4, 3, 3))
    dX = np.zeros(X.shape)
    # sample
    for i in range(2):
        x = X[i]
        x = np.pad(x, ((0,0),(pad,pad),(pad,pad)), "constant")
        # output
        for d in range(2):
            # channels
            for c in range(4):
                for h in range(NH):
                    for w in range(NW):
                        # ty[i,d,h,w]
                        for fh in range(3):
                            for fw in range(3):
                                # ty[i,d,h,w] += sum ~ F[d, c, fh, fw] * X[i, c, h*s + fh, w*s + fw]
                                ph = h * stride + fh
                                pw = w * stride + fw
                                dF[d, c, fh, fw] += conv.dY[i, d, h, w] * x[c, ph, pw]
                                oh = ph - pad
                                ow = pw - pad
                                if oh >= 0 and ow >= 0 and oh < 10 and ow < 10:
                                    dX[i, c, oh, ow] += conv.dY[i, d, h, w] * F[d, c, fh, fw]

    dF /= 2
    assert np.allclose(conv.db, db.reshape(conv.db.shape))
    assert np.allclose(conv.dW, dF.reshape(conv.dW.shape))
    assert np.allclose(conv.dX, dX)


def test_conv():
    go_conv(1, 0)
    go_conv(2, 0)
    go_conv(3, 0)
    go_conv(1, 1)
    go_conv(2, 1)
    go_conv(3, 1)
