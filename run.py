import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import gc
import sys

from load_data import load_split_data, load_set_data, load_test_frames, load_frames_data
from input_distortion import input_distortion,pretraining_input_distortion
from copy import copy

from MAE import MAE

# Load data for training
print('load data')
train_seq, val_seq, test_seq = load_split_data() 

print('generate training and validation data')
data_train = load_set_data(train_seq)
data_val = load_set_data(val_seq)

print('initialize model')
# Define Full Model
mae = MAE()

# Train MultiModal AutoEncoder
print('start training')
mae.train_model(data_train,data_val,load = 'pretrained')
del data_train, data_val
