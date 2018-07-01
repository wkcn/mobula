import mobula.layers as L
import numpy as np

def go_conv(stride, pad):
    X = np.random.random((2, 4, 10, 10)) * 100
    N, C, H, W = X.shape
    K = 3
    D = 2
    FW = np.random.random((D, C, K * K)) * 10
    F = FW.reshape((D, C, K, K))

    data = L.Data(X)
    conv = L.Conv(data, kernel = K, pad = pad, stride = stride, dim_out = D)

    pad_h = pad_w = pad
    kernel_h = kernel_w = K

    XH = H
    XW = W

    NH = (XH + pad_h * 2 - kernel_h) // stride + 1
    NW = (XW + pad_w * 2 - kernel_w) // stride + 1

    data.reshape()
    conv.reshape()

    conv.W = FW
    conv.b = np.random.random(conv.b.shape) * 10

    ty = np.zeros((N, D, NH, NW))
    for i in range(N):
        x = X[i]
        x = np.pad(x, ((0,0),(pad_h,pad_h),(pad_w,pad_w)), "constant")
        for d in range(D):
            f = F[d] # 4, 3, 3
            for h in range(NH):
                for w in range(NW): 
                    th = h * stride
                    tw = w * stride
                    a = x[:, th:th+K, tw:tw+K]
                    e = np.sum(a * f)
                    ty[i, d, h, w] = e
        ty[i] += conv.b.reshape((D, 1, 1))
    conv.forward()
    print (np.max(np.abs(conv.Y - ty)), conv.Y.shape)
    assert np.allclose(conv.Y, ty)

    conv.dY = np.random.random(conv.Y.shape) * 50
    # test backward, dX, dW, db
    conv.backward()

    db = np.sum(conv.dY, (0, 2, 3)).reshape(conv.b.shape)

    dF = np.zeros((D, C, K, K))
    dX = np.zeros(X.shape)
    # sample
    for i in range(N):
        x = X[i]
        x = np.pad(x, ((0,0),(pad_h,pad_h),(pad_w,pad_w)), "constant")
        # output
        for d in range(D):
            # channels
            for c in range(C):
                for h in range(NH):
                    for w in range(NW):
                        # ty[i,d,h,w]
                        for fh in range(K):
                            for fw in range(K):
                                # ty[i,d,h,w] += sum ~ F[d, c, fh, fw] * X[i, c, h*s + fh, w*s + fw]
                                ph = h * stride + fh
                                pw = w * stride + fw
                                dF[d, c, fh, fw] += conv.dY[i, d, h, w] * x[c, ph, pw]
                                oh = ph - pad_h
                                ow = pw - pad_w
                                if oh >= 0 and ow >= 0 and oh < H and ow < W:
                                    dX[i, c, oh, ow] += conv.dY[i, d, h, w] * F[d, c, fh, fw]

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
