#coding=utf-8
import numpy as np

def Xavier(shape):
    # shape: (dim_out, dim_in)
    R = np.random.random(shape)
    k = np.sqrt(6.0 / (shape[0] + shape[1]))
    return -k + (2 * k) * R
