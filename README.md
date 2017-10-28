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

for first 10000 test data frames:

1.with RGBDS full input       

RMSE error is 4.633(depth<50m)/7.133(depth<80m)  
absr error is 0.143 /0.161
valid depth points used in measurement 900 per 60*18 image.   


2.with only RGB input   


.....
 


