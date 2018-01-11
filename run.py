from load_data import load_split_data, load_set_data, load_test_frames, load_frames_data
from RecurrentMAE import RecurrentMAE
from MAE import MAE

# Load data for training
print('load data')
train_seq, val_seq, test_seq = load_split_data()



print('generate training and validation data')
data_train = load_set_data(train_seq)
data_val = load_set_data(val_seq)

test_frames = load_test_frames()
data_test = load_frames_data(test_frames)

print('initialize model')
# Define Full Model
'''
mae = MAE(n_epochs=150,
          learning_rate=1e-4,
          mirroring=True,
          verbose=True)

'''
rnn_mae = RecurrentMAE(n_epochs=400,
                       rnn_option='lstm',
                       n_rnn_steps=5,
                       mirroring=False,
                       learning_rate=1e-3,
                       sharing='nonshared',
                       load_previous=False,
                       model='new')


# Train MultiModal AutoEncoder
print('start training')
rnn_mae.train_model(data_train,data_val)
#rnn_mae.train_model(data_train,data_val)
#del data_train, data_val


#mae.evaluate(data_test,run='20171126-122501')
