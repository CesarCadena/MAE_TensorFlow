#import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import gc
import sys
sys.path.append('./tools')

from load_data import load_split_data, load_set_data, load_test_frames, load_frames_data
from input_distortion import input_distortion,pretraining_input_distortion
from copy import copy

from MAE import MAE

# Load data for training
train_seq, val_seq, test_seq = load_split_data() 

data_train = load_set_data(train_seq)
data_val = load_set_data(val_seq)

# Define Full Model
mae = MAE()

# Train MultiModal AutoEncoder
mae.train_model(data_train,data_val,load = 'pretrained')
del data_train, data_val
