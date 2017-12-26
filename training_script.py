from load_data import load_split_data, load_set_data, load_test_frames, load_frames_data
from RecurrentMAE import RecurrentMAE

# Load data for training
print('load data')
train_seq, val_seq, test_seq = load_split_data()

print('generate training and validation data')
data_train = load_set_data(train_seq)
data_val = load_set_data(val_seq)


# basic rnn mae
rnn_mae = RecurrentMAE(n_epochs=1,rnn_option='basic',n_rnn_steps=5,mirroring=False,learning_rate=1e-06,
                       load_previous=False)
rnn_mae.train_model(data_train,data_val)

rnn_mae = RecurrentMAE(n_epochs=1,rnn_option='basic',n_rnn_steps=7,mirroring=False,learning_rate=1e-06,
                       load_previous=True)
rnn_mae.train_model(data_train,data_val)

rnn_mae = RecurrentMAE(n_epochs=1,rnn_option='basic',n_rnn_steps=10,mirroring=False,learning_rate=1e-06,
                       load_previous=True)
rnn_mae.train_model(data_train,data_val)

rnn_mae = RecurrentMAE(n_epochs=1,rnn_option='basic',n_rnn_steps=12,mirroring=False,learning_rate=1e-06,
                       load_previous=True)
rnn_mae.train_model(data_train,data_val)

rnn_mae = RecurrentMAE(n_epochs=1,rnn_option='basic',n_rnn_steps=15,mirroring=False,learning_rate=1e-06,
                       load_previous=True)
rnn_mae.train_model(data_train,data_val)

'''

# lstm rnn mae
rnn_mae = RecurrentMAE(n_epochs=1000,rnn_option='lstm',n_rnn_steps=5,mirroring=False,learning_rate=1e-06,
                       load_previous=False)
rnn_mae.train_model(data_train,data_val)

rnn_mae = RecurrentMAE(n_epochs=1000,rnn_option='lstm',n_rnn_steps=7,mirroring=False,learning_rate=1e-06,
                       load_previous=True)
rnn_mae.train_model(data_train,data_val)

rnn_mae = RecurrentMAE(n_epochs=1000,rnn_option='lstm',n_rnn_steps=10,mirroring=False,learning_rate=1e-06,
                       load_previous=True)
rnn_mae.train_model(data_train,data_val)

rnn_mae = RecurrentMAE(n_epochs=1000,rnn_option='lstm',n_rnn_steps=12,mirroring=False,learning_rate=1e-06,
                       load_previous=True)
rnn_mae.train_model(data_train,data_val)

rnn_mae = RecurrentMAE(n_epochs=1000,rnn_option='lstm',n_rnn_steps=15,mirroring=False,learning_rate=1e-06,
                       load_previous=True)
rnn_mae.train_model(data_train,data_val)

# gated rnn mae
rnn_mae = RecurrentMAE(n_epochs=1000,rnn_option='gated',n_rnn_steps=5,mirroring=False,learning_rate=1e-06,
                       load_previous=False)
rnn_mae.train_model(data_train,data_val)

rnn_mae = RecurrentMAE(n_epochs=1000,rnn_option='gated',n_rnn_steps=7,mirroring=False,learning_rate=1e-06,
                       load_previous=True)
rnn_mae.train_model(data_train,data_val)

rnn_mae = RecurrentMAE(n_epochs=1000,rnn_option='gated',n_rnn_steps=10,mirroring=False,learning_rate=1e-06,
                       load_previous=True)
rnn_mae.train_model(data_train,data_val)

rnn_mae = RecurrentMAE(n_epochs=1000,rnn_option='gated',n_rnn_steps=12,mirroring=False,learning_rate=1e-06,
                       load_previous=True)
rnn_mae.train_model(data_train,data_val)

rnn_mae = RecurrentMAE(n_epochs=1000,rnn_option='gated',n_rnn_steps=15,mirroring=False,learning_rate=1e-06,
                       load_previous=True)
rnn_mae.train_model(data_train,data_val)

'''




