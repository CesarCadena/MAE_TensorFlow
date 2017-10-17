import numpy as np

from copy import copy

def rms_error(xest,xgt):


    xest = np.reshape(xest,(max(xest.shape),1))
    xgt = np.reshape(xgt,(max(xgt.shape),1))

    T1 = np.zeros(xest.shape)
    T2 = np.zeros(xgt.shape)

    for i in range(0,xest.shape[0]):
        if xest[i]>0.02:
            T1[i] = 1
        if xgt[i]>0.02:
            T2[i] = 1

    T3 = T1+T2
    T = np.where(T3==2)[0]

    xest = 1./xest[T]
    xgt = 1./xgt[T]


    diff = xest - xgt
    e = np.sqrt(np.sum(np.square(diff))/len(T))
    return e

def relative_error(xest,xgt):

    xest = np.reshape(xest,(max(xest.shape),1))
    xgt = np.reshape(xgt,(max(xgt.shape),1))

    T1 = np.zeros(xest.shape)
    T2 = np.zeros(xgt.shape)

    for i in range(0,xest.shape[0]):
        if xest[i]>0.02:
            T1[i] = 1
        if xgt[i]>0.02:
            T2[i] = 1

    T3 = T1+T2
    T = np.where(T3==2)[0]

    xest = 1./xest[T]
    xgt = 1./xgt[T]

    diff = xest-xgt
    s = np.divide(np.abs(diff),xgt)
    e = np.sum(s)/xgt.shape[0]
    return e
