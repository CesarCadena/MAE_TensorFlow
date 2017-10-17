import numpy as np

def invert_depth(inv_depth):

    inv_depth = np.reshape(inv_depth,(max(inv_depth.shape),1))
    depth = np.zeros(inv_depth.shape)

    for i in range(0,inv_depth.shape[0]):
        if inv_depth[i] > 0.02:
            depth[i] = 1./inv_depth[i]
    return depth
