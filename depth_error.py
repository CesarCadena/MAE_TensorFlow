# coding: utf-8
import numpy as np
def ABSR(true,pred):

    T1=truth>1e-8
    T2=prediction>1e-8
    T=np.logical_and(T1,T2)
    truth=truth[T]
    prediction=prediction[T]

    T3=truth<50
    T4=prediction<50
    T=np.logical_and(T3,T4)
    prediction=prediction[T]
    truth=truth[T]

    n=len(truth)
    error=sum(abs(truth-prediction)/truth)
    abs_error=error/n
    
    return abs_error


def inverse_depth(a):
    b=1.0/(a+1e-8)
    return b

def RMSE(truth,prediction):

    T1=truth>1e-8
    T2=prediction>1e-8
    T=np.logical_and(T1,T2)
    truth=truth[T]
    prediction=prediction[T]

    T3=truth<50
    T4=prediction<50
    T=np.logical_and(T3,T4)
    prediction=prediction[T]
    truth=truth[T]

    n=len(truth)
    error=sum((truth-prediction)**2)/n
    rmse_error=np.sqrt(error)
    
    return rmse_error
