import numpy as np

def rms_error(xest,xgt):

    xest = np.reshape(xest,(max(xest.shape),1))
    xgt = np.reshape(xgt,(max(xgt.shape),1))

    T = np.where(xest > 0)[0]
    T = np.where(xgt[T] > 0)[0]

    diff = xest[T] - xgt[T]
    e = np.sqrt(np.sum(np.square(diff))/len(T))
    return e

def relative_error(xest,xgt):
    T = np.where(xest>0)[0]
    T = np.where(xgt[T]>0)[0]
    diff = xest[T] - xgt[T]
    s = np.divide(np.abs(diff),xgt[T])
    e = np.sum(s)/len(T)
    return e
