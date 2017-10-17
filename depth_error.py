# coding: utf-8
import numpy as np
def ABSR(true,pred):

    T=(np.where(true>0) and np.where(pred>0) )[0]

    true=true[T]
    pred=pred[T]
    n=len(T)


   
    error=abs(true-pred)
    r_error=error/true
    
    r_error=sum(r_error)/n
    
    return r_error


def RMSE(truth,prediction):

    T1=truth>0
    T2=prediction>0
    T=np.logical_and(T1,T2)

    truth=truth[T]
    prediction=prediction[T]
    print(len(prediction))
    truth=1/truth
    prediction=1/prediction

    T3=prediction<50
    prediction=prediction[T3]
    truth=truth[T3]
    
    n=len(truth)
    print(n)
  
    error=sum((truth-prediction)**2)
    rmse_error=np.sqrt(error/n)
    
    return rmse_error

