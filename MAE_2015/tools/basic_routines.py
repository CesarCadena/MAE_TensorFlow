import numpy as np
from copy import copy

def invert_depth(inv_depth):

    inv_depth = np.reshape(inv_depth,(max(inv_depth.shape),1))
    depth = np.zeros(inv_depth.shape)

    for i in range(0,inv_depth.shape[0]):
        if inv_depth[i] > 1/80.:
            depth[i] = 1./inv_depth[i]
    return depth

def horizontal_mirroring(x,ind_who,resolution = (18,60)):
    
    h = resolution[0]
    w = resolution[1]
    indmirror = np.linspace(0,h*w-1,h*w).astype(int)
    indmirror = np.reshape(indmirror,(w,h))
    indmirror = indmirror.T
    indmirror = np.fliplr(indmirror)
    indmirror = indmirror.T
    indmirror = np.reshape(indmirror,(h*w))
    
    x1 = copy(x)
    x = np.array(x)
    for i in ind_who:
        x1[i]=x[i][indmirror]
    
    return x1

def zeroing_channel(x,ind_who,resolution = (18,60)):
    
    h = resolution[0]
    w = resolution[1]
        
    x1 = copy(x)    
    zeros = np.zeros((h*w))
    
    for i in ind_who:
        x1[i] = zeros
    
    return x1