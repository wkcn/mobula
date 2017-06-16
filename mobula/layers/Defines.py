#coding=utf-8
import numpy as np

def Xavier(shape):
    # shape: (dim_out, dim_in)
    R = np.random.random(shape)
    k = np.sqrt(6.0 / R.size)
    return -k + (2 * k) * R


def im2col(A, BSZ, stepsize = 1):
    # Parameters
    m,n = A.shape
    s0, s1 = A.strides    
    nrows = m-BSZ[0]+1
    ncols = n-BSZ[1]+1
    shp = BSZ[0],BSZ[1],nrows,ncols
    strd = s0,s1,s0,s1

    out_view = np.lib.stride_tricks.as_strided(A, shape=shp, strides=strd)
    return out_view.reshape(BSZ[0]*BSZ[1],-1)[:,::stepsize]
