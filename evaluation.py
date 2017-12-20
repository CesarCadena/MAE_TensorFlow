import numpy as np
import tensorflow as tf
import evaluation_functions

import matplotlib.pyplot as plt
from load_data import load_test_data, load_frames_data, load_training_data, load_validation_data, load_test_frames, load_test_sequences
import RecurrentMAE
from MAE import MAE


from build_test_sequences import build_test_sequences

data_test = load_test_data()
data_test = build_test_sequences(data_test,n_rnn_steps=5)

seq = 2

sequence = [data_test[0][seq],
            data_test[1][seq],
            data_test[2][seq],
            data_test[3][seq],
            data_test[5][seq],
            data_test[6][seq],
            data_test[7][seq],
            data_test[8][seq],
            data_test[9][seq]]

# running model
run = '20171218-153528'
mae = MAE(n_epochs=1000,learning_rate=1e-04,mirroring=True)
rms1,rel1 = mae.evaluate_sequence(sequence,option='rgbd',n_rnn_steps=5,frequency=5,run=run)

del mae


run = '20171204-173250' # needs to be chosen

RNN_MAE = RecurrentMAE.RecurrentMAE(n_epochs=1000,rnn_option='basic',n_rnn_steps=5,mirroring=False,learning_rate=1e-07)


rms2,rel2 = RNN_MAE.evaluate_sequence(sequence,n_rnn_steps=5,option='rgbd',frequency=5,run=run)


steps = np.arange(0,len(rms1)).astype(int)

rms = [rms1,rms2]
rms = np.asarray(rms)
rms = np.reshape(rms,(rms.shape[1],rms.shape[0]))


plt.plot(steps,rms)
plt.grid()
plt.xlabel('Steps')
plt.ylabel('RMS Error')
plt.show()
