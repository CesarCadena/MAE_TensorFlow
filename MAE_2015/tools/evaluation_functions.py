import numpy as np

from copy import copy

def rms_error(xest,xgt):

    xest = np.reshape(xest,(max(xest.shape),1))
    xgt = np.reshape(xgt,(max(xgt.shape),1))

    T1 = np.ones(xest.shape)
    T2 = np.ones(xgt.shape)

    for i in range(0,xest.shape[0]):
        if xest[i] == 0:
            T1[i] = 0
        if xgt[i] == 0:
            T2[i] = 0

    T3 = T1+T2
    T = np.where(T3==2)[0]

    diff = xest[T] - xgt[T]
    e = np.sqrt(np.sum(np.square(diff))/len(T))
    return e

def relative_error(xest,xgt):

    xest = np.reshape(xest,(max(xest.shape),1))
    xgt = np.reshape(xgt,(max(xgt.shape),1))

    T1 = np.ones(xest.shape)
    T2 = np.ones(xgt.shape)

    for i in range(0,xest.shape[0]):
        if xest[i] == 0:
            T1[i] = 0
        if xgt[i] == 0:
            T2[i] = 0

    T3 = T1+T2
    T = np.where(T3==2)[0]

    diff = xest[T]-xgt[T]
    s = np.divide(np.abs(diff),xgt[T])
    e = np.sum(s)/len(T)
    return e


def inter_union(xest,xgt):

    #xest = np.reshape(xest,(max(xest.shape),1))
    #xgt = np.reshape(xgt,(max(xgt.shape),1))
    void = 0.5*np.ones([1,xgt.shape[1],xgt.shape[2]])
    
    predicted_class = np.argmax(xest,axis=0)
    gt_class = np.argmax(np.concatenate((xgt,void),axis=0),axis=0)
    
    inter = np.zeros(xgt.shape[0])
    union = np.zeros(xgt.shape[0])
                      
    for i in range(0,xgt.shape[0]):
        inter[i] = np.count_nonzero(((gt_class == i) | (gt_class == xgt.shape[0])) & (predicted_class == i))
        union[i] = np.count_nonzero((gt_class == i) | (predicted_class == i))


    iu = inter/union
    return iu, inter, union


