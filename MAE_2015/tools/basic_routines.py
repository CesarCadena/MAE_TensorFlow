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

def color2image(red_channel,green_channel,blue_channel,resolution=(18,60)):
    
    normalization = 1.
    if max(red_channel) > 1.:
        normalization = 255.
        
    imr = np.reshape(red_channel/normalization,(resolution[1],resolution[0])).T
    img = np.reshape(green_channel/normalization,(resolution[1],resolution[0])).T
    imb = np.reshape(blue_channel/normalization,(resolution[1],resolution[0])).T
    im = np.dstack((imr,img,imb))
    
    return im

def depth2image(depth_channel,mask_channel,resolution=(18,60)):
    
    imd = np.reshape(depth_channel,(resolution[1],resolution[0])).T * np.reshape(mask_channel,(resolution[1],resolution[0])).T
    
    return imd

def sem2image(semantic_channel,resolution=(18,60)):

    ims = np.reshape(semantic_channel,(resolution[1],resolution[0])).T
    
    return ims