import numpy as np

from copy import copy

def rms_error(xest,xgt):


    xest = np.reshape(xest,(max(xest.shape),1))
    xgt = np.reshape(xgt,(max(xgt.shape),1))

    T = (xgt <= 80.) & (xgt > 0.0)

    print(T)
    print(len(T))

    diff = xest[T] - xgt[T]
    e = np.sqrt(np.sum(np.square(diff))/len(T))
    return e

def relative_error(xest,xgt):

    xest = np.reshape(xest,(max(xest.shape),1))
    xgt = np.reshape(xgt,(max(xgt.shape),1))

    T = (xgt <= 80.) & (xgt > 0.0)

    print(len(T))

    diff = xest[T]-xgt[T]
    s = np.divide(np.abs(diff),xgt[T])
    e = np.sum(s)/len(T)
    return e



