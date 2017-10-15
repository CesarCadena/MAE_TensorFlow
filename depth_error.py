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


def RMSE(true,pred):
    T=(np.where(true>1e-8) and np.where(pred>1e-8) )[0]
    
    
    true=true[T]
    pred=pred[T]
 
    n=len(T)
  
    error=sum((true-pred)**2)
    rmse_error=np.sqrt(error/n)
    
    return rmse_error

