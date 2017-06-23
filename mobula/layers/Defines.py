#coding=utf-8
import numpy as np

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

def get_idx_from_arg(a, idx, axis):
    shp = a.shape
    dim_idx = list(np.ix_(*[np.arange(shp[i]) for i in range(len(shp)) if i != axis]))
    dim_idx.append(idx)
    return dim_idx

def get_val_from_arg(a, idx, axis):
    '''
        idx = np.argmax(a, 1)
        get_val_from_arg(a, idx, axis = 1) is equal to np.max(a, 1) 
    '''
    dim_idx = get_idx_from_arg(a, idx, axis)
    return a.swapaxes(axis, -1)[dim_idx]
