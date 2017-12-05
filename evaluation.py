import numpy as np
import tensorflow as tf
import evaluation_functions
from load_data import load_test_data, load_frames_data, load_training_data, load_validation_data, load_test_frames, load_test_sequences
import RecurrentMAE

test_frames = load_test_frames()
data_test = load_test_sequences(test_frames,n_steps=5)


# running model

run = '20171204-173250' # needs to be chosen

RNN_MAE = RecurrentMAE.RecurrentMAE(rnn_option='basic',n_rnn_steps=5)
RNN_MAE.evaluate(data_test,run=run)
