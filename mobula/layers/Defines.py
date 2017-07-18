#coding=utf-8
import numpy as np
import numpy_groupies as npg

def Xavier(shape):
    # shape: (dim_out, dim_in)
    R = np.random.random(shape)
    k = np.sqrt(6.0 / R.size)
    return -k + (2 * k) * R


def im2col(A, fshape, stride = 1):
    h, w = A.shape
    fh, fw = fshape
    s0, s1 = A.strides
    rows = h - fh + 1
    cols = w - fw + 1
    nw = (w - fw) // stride + 1
    nh = (h - fh) // stride + 1
    shp = fh, fw, rows, cols
    strd = s0, s1, s0, s1
    view = np.lib.stride_tricks.as_strided(A, shape = shp, strides = strd)
    i = np.tile(np.arange(0, rows, stride), nh) + np.repeat(np.arange(nh) * (rows * stride), nw)
    return view.reshape(fh * fw, -1)[:, i]


def get_idx_from_arg(a, arg, axis):
    shp = a.shape
    cp = np.cumprod(shp[::-1])[::-1]
    if axis == len(shp) - 1:
        m = 1
    else:
        m = cp[axis + 1]
    n = cp[0] // cp[axis]
    return np.repeat(np.arange(n) * cp[axis], m) + np.tile(np.arange(m), n) + arg.ravel() * m 

def get_val_from_idx(a, idx):
    return a.ravel()[idx]
