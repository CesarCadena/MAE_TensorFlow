import numpy as np

def rms_error(xest,xgt):
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


a = np.ones((10,1))
b1 = 0.1*np.ones((5,1))
b2 = np.ones((5,1))
b = np.vstack((b1,b2))


e = rms_error(a,b)
e = relative_error(a,b)
print(e)
