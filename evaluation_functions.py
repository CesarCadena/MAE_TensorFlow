import numpy as np

from copy import copy

def rms_error(xest,xgt):


    xest = np.reshape(xest,(max(xest.shape),1))
    xgt = np.reshape(xgt,(max(xgt.shape),1))

    T = np.where(xgt < 80)

    print(len(T))

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

    print(len(T))

    diff = xest[T]-xgt[T]
    s = np.divide(np.abs(diff),xgt[T])
    e = np.sum(s)/len(T)
    return e



