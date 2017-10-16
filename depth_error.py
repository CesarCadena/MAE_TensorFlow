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
    T3=prediction>50
    T=np.logical_and(T1,T2)
    T=np.logical_and(T,T3)


    #print(T)
    truth=truth[T]
    truth=1/truth
    prediction=prediction[T]
    prediction=1/prediction
    
    n=sum(T)
    #print(n)
  
    error=sum((truth-prediction)**2)
    rmse_error=np.sqrt(error/n)
    
    return rmse_error

