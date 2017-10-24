# coding: utf-8
import numpy as np
def ABSR(truth,prediction):

    T1=truth>0.02
    T2=prediction>0.02
    T=np.logical_and(T1,T2)
    truth=inverse_depth(truth[T])
    prediction=inverse_depth(prediction[T])

    #T3=truth<50
    #T4=prediction<50
    #T=np.logical_and(T3,T4)
    #prediction=prediction[T]
    #truth=truth[T]

    n=len(truth)
    error=sum(abs(truth-prediction)/truth)
    abs_error=error/n
    
    return abs_error


def inverse_depth(a):
    b=1.0/a
    return b

def RMSE(truth,prediction):

    T1=truth>0.02
    T2=prediction>0.02
    T=np.logical_and(T1,T2)
    truth=inverse_depth(truth[T])
    prediction=inverse_depth(prediction[T])

    #T3=truth<50
    #T4=prediction<50
    #T=np.logical_and(T3,T4)
    #prediction=prediction[T]
    #truth=truth[T]

    n=len(truth)
    error=sum((truth-prediction)**2)/n
    rmse_error=np.sqrt(error)
    
    return rmse_error
