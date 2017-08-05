import mobula.layers as L
import numpy as np

def test_contrastive():
    A = np.array([[1,2,0], [2,1,0]]).astype(np.float)
    B = np.array([[0,1,3], [2,3,1]]).astype(np.float)

    diff = A - B # 2 x 3
    dist_sq = np.sum(np.square(diff), 1).reshape((2,1)) # 2 x 1
    dist = np.sqrt(dist_sq) # 2 x 1

    print ("diff", diff)

    for i in range(2):
        for j in range(2):
            sim = np.array([i, j]).reshape((2, 1))


            data = L.Data([A,B,sim])
            [a,b,s] = data()

            l = L.ContrastiveLoss([a,b,s], "ctr", margin = 6.0)

            print ("sim", s.Y.ravel())
            l.forward()
            l.backward()

            # margin - d
            md = l.margin - dist
            # max(md, 0)
            ma = np.clip(md, 0, np.inf)

            same = (s.Y == 1)

            print ("sha", dist_sq.shape, same.shape)
            sa = dist_sq * same
            sb = np.square(ma) * (1 - same)
            print ("lr", sa, sb)
            target = np.sum(sa + sb) / 2.0 / 2
            print ("t", target, l.Y, target - l.Y)
            assert np.allclose(target, l.Y)

            left = diff / 2.0
            right = -ma / 2.0 * diff / dist
            w = left * same + right * (1.0 - same)
            print ("dx", l.dX[0], w, l.dX[0] - w)
            assert np.allclose(l.dX[0], w)
