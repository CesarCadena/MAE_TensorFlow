#import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import gc
import sys
sys.path.append('./tools')

from load_data import load_split_data, load_set_data, load_test_frames, load_frames_data
from input_distortion import input_distortion,pretraining_input_distortion
from copy import copy

from PretrainingMAE import PretrainingMAE

# Load data for training
train_seq, val_seq, test_seq = load_split_data() 

data_train = load_set_data(train_seq)
data_val = load_set_data(val_seq)

# Define models for Pretraining
pretraining =  PretrainingMAE(data_train,data_val)
del data_train, data_val

# Pretraing RGB Channels independently
pretraining.pretrain_red_channel()
pretraining.pretrain_green_channel()
pretraining.pretrain_blue_channel()

# Pretraing Inverse-Depth Channel
pretraining.pretrain_depth_channel()

# Pretraing Semantic Channels independently
pretraining.pretrain_gnd_channel()
pretraining.pretrain_obj_channel()
pretraining.pretrain_bld_channel()
pretraining.pretrain_veg_channel()
pretraining.pretrain_sky_channel()

# Pretraing Semantic AutoEncoder
pretraining.pretrain_shared_semantics(run=pretraining.run)
