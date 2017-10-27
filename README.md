# README #

Multimodal Autoencoders on TensorFlow.

The data for training, validation and testing can be found here:

https://polybox.ethz.ch/index.php/s/CNaOah6Ydq7WrRK



# How to run  ...
1.run MAE_pretraining_sep.py  
2.run MAE_pretraining_sem_shared.py  
3.run fullMAE.py  

# load test data   ...  
1.run MAE_test.py  

# depth evaluation

for first 1000 test data frames ,  
with RGBDs input   
RMSE error is 9.480
absr error is 0.3125
processing time for one frame is 20ms on mac laptop.  


