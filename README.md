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

for first 10000 test data frames ,  
with RGBDs full input     
RMSE error is 8.5547  
absr error is 0.3103  
valid depth points used in measurement 700~800 per 60*18 image.  

with only RGB input  

.....
 


