import mobula.layers as L
import numpy as np

def go_pool(stride, kernel, kind):
    print ("test_pool", stride, kernel, kind)
    X = np.random.random((2, 3, 10, 10)) * 100
    N, C, H, W = X.shape

    data = L.Data(X, "data")
    pool = L.Pool(data, "Pool", kernel = kernel, stride = stride, pool = kind)

    kernel_h = kernel_w = kernel

    NH = (H - kernel_h) // stride + 1
    NW = (W - kernel_w) // stride + 1

    data.reshape()
    pool.reshape()

    ty = np.zeros((N, C, NH, NW))
    tyi = np.zeros(ty.shape)
    for i in range(N):
        x = X[i]
        for d in range(C):
            for h in range(NH):
                for w in range(NW): 
                    th = h * stride
                    tw = w * stride
                    a = x[d, th:th+kernel, tw:tw+kernel]
                    mi = -1

                    if kind == L.Pool.MAX:
                        e = np.max(a)
                        mi = np.argmax(a)
                    else:
                        # avg
                        e = np.mean(a)

                    ty[i, d, h, w] = e
                    tyi[i, d, h, w] = mi

    pool.forward()
    assert np.allclose(pool.Y, ty)

    pool.dY = np.random.random(pool.Y.shape) * 39

    dX = np.zeros(pool.X.shape)
    if kind == L.Pool.MAX:
        for i in range(N):
            for d in range(C):
                for h in range(NH):
                    for w in range(NW): 
                        mi = tyi[i, d, h, w]
                        th = h * stride
                        tw = w * stride
                        a = dX[i, d, th:th+kernel, tw:tw+kernel]
                        a[int(mi) // kernel, int(mi) % kernel] += pool.dY[i, d, h, w] 
    else:
        for i in range(N):
            for d in range(C):
                for h in range(NH):
                    for w in range(NW): 
                        mi = tyi[i, d, h, w]
                        th = h * stride
                        tw = w * stride
                        dX[i, d, th:th+kernel, tw:tw+kernel] += pool.dY[i, d, h, w] / (kernel * kernel)
                
    pool.backward()
    assert np.allclose(pool.dX, dX)


def test_pool():
    for kernel in [2, 3, 5, 10]: 
        for stride in [1, 2, 3]:
            for kind in [L.Pool.MAX, L.Pool.AVG]:
                go_pool(stride, kernel, kind)
