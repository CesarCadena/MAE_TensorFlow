import scipy.io as io
import scipy.misc as misc
import os
import numpy as np
import matplotlib.pyplot as plt


def load_data():
    with open('../03-data/MAE_KITTI/kitti_split.txt') as file:
        datatext = file.readlines()

    data_train = []
    for i in datatext:
        if i == 'Test:\n':
            break
        if i == 'Train:\n':
            continue
        if i == '\n':
            continue
        data_train.append(i.replace('\n',''))


    datatext = datatext[len(data_train)+4::]

    data_test = []
    for i in datatext:
        if i == 'Test:\n':
            continue
        if i == 'Val:\n':
            break
        if i == '\n':
            continue
        data_test.append(i.replace('\n',''))

    datatext = datatext[len(data_test)+3::]

    data_val = []
    for i in datatext:
        if i == 'Val:\n':
            continue
        if i == '\n':
            continue
        data_val.append(i.replace('\n',''))

    basedir = '../03-data/MAE_KITTI/data_18x60/'
    basefile = 'data_kitti_'
    filetypes = ['im02','im03','InvDepth02','InvDepth03','seg02','seg03']

    data_training = []
    for i in filetypes:
        for j in data_train:
            data_training.append(io.loadmat(basedir+basefile+i+'_'+j+'_18x60.mat'))

    data_testing = []
    for i in filetypes:
        for j in data_test:
            data_testing.append(io.loadmat(basedir+basefile+i+'_'+j+'_18x60.mat'))

    data_validation = []
    for i in filetypes:
        for j in data_val:
            data_validation.append(io.loadmat(basedir+basefile+i+'_'+j+'_18x60.mat'))



    # PREPARING TRAINING DATA

    width = 60
    height = 18

    im_shape = (width,height)

    n_training = len(data_train)
    n_categories = int(len(data_training)/n_training)


    training = []
    frame =  {'xcr1': 0,
              'xcg1': 0,
              'xcb1': 0,
              'xcr2': 0,
              'xcg2': 0,
              'xcb2': 0,
              'xidspars1': 0,
              'xid1': 0,
              'xmaks1': 0,
              'xidsparse2': 0,
              'xid2': 0,
              'xmask2': 0,
              'sem1':0,
              'sem2':0}

    for i in range(0,n_training):
        seq = []
        for j in range(0,data_training[i]['xcr'].shape[1]):
            frame['xcr1'] = data_training[i]['xcr'][:,j]
            frame['xcg1'] = data_training[i]['xcg'][:,j]
            frame['xcb1'] = data_training[i]['xcb'][:,j]

            frame['xcr2'] = data_training[i+n_training]['xcr'][:,j]
            frame['xcg2'] = data_training[i+n_training]['xcg'][:,j]
            frame['xcb2'] = data_training[i+n_training]['xcb'][:,j]

            frame['xidsparse1'] = data_training[i+2*n_training]['xidsparse'][:,j]
            frame['xid1'] = data_training[i+2*n_training]['xid'][:,j]
            frame['xmask1'] = data_training[i+2*n_training]['xmask'][:,j]

            frame['xidsparse2'] = data_training[i+3*n_training]['xidsparse'][:,j]
            frame['xid2'] = data_training[i+3*n_training]['xid'][:,j]
            frame['xmask2'] = data_training[i+3*n_training]['xmask'][:,j]

            frame['sem1'] = data_training[i+4*n_training]['xss'][:,j]
            frame['sem2'] = data_training[i+5*n_training]['xss'][:,j]

            seq.append(frame)
        training.append(seq)


    validation = []
    n_validation = len(data_val)

    for i in range(0,n_validation):
        seq = []

        for j in range(0,data_validation[i]['xcr'].shape[1]):
            frame['xcr1'] = data_validation[i]['xcr'][:,j]
            frame['xcg1'] = data_validation[i]['xcg'][:,j]
            frame['xcb1'] = data_validation[i]['xcb'][:,j]

            frame['xcr2'] = data_validation[i+n_validation]['xcr'][:,j]
            frame['xcg2'] = data_validation[i+n_validation]['xcg'][:,j]
            frame['xcb2'] = data_validation[i+n_validation]['xcb'][:,j]

            frame['xidsparse1'] = data_validation[i+2*n_validation]['xidsparse'][:,j]
            frame['xid1'] = data_validation[i+2*n_validation]['xid'][:,j]
            frame['xmask1'] = data_validation[i+2*n_validation]['xmask'][:,j]

            frame['xidsparse2'] = data_validation[i+3*n_validation]['xidsparse'][:,j]
            frame['xid2'] = data_validation[i+3*n_validation]['xid'][:,j]
            frame['xmask2'] = data_validation[i+3*n_validation]['xmask'][:,j]

            frame['sem1'] = data_validation[i+4*n_validation]['xss'][:,j]
            frame['sem2'] = data_validation[i+5*n_validation]['xss'][:,j]

            seq.append(frame)

        validation.append(seq)


    testing = []
    n_testing = len(data_test)

    for i in range(0,n_testing):
        seq = []

        seq_length = []

        seq_length.append(data_testing[i+1]['xcr'].shape[1])
        seq_length.append(data_testing[i+n_testing]['xcr'].shape[1])
        seq_length.append(data_testing[i+2*n_testing]['xidsparse'].shape[1])
        seq_length.append(data_testing[i+3*n_testing]['xidsparse'].shape[1])
        seq_length.append(data_testing[i+4*n_testing]['xss'].shape[1])
        seq_length.append(data_testing[i+5*n_testing]['xss'].shape[1])

        min_seq_length = min(seq_length)
        for j in range(0,min_seq_length):
            frame['xcr1'] = data_testing[i]['xcr'][:,j]
            frame['xcg1'] = data_testing[i]['xcg'][:,j]
            frame['xcb1'] = data_testing[i]['xcb'][:,j]

            frame['xcr2'] = data_testing[i+n_testing]['xcr'][:,j]
            frame['xcg2'] = data_testing[i+n_testing]['xcg'][:,j]
            frame['xcb2'] = data_testing[i+n_testing]['xcb'][:,j]

            frame['xidsparse1'] = data_testing[i+2*n_testing]['xidsparse'][:,j]
            frame['xid1'] = data_testing[i+2*n_testing]['xid'][:,j]
            frame['xmask1'] = data_testing[i+2*n_testing]['xmask'][:,j]

            frame['xidsparse2'] = data_testing[i+3*n_testing]['xidsparse'][:,j]
            frame['xid2'] = data_testing[i+3*n_testing]['xid'][:,j]
            frame['xmask2'] = data_testing[i+3*n_testing]['xmask'][:,j]

            frame['sem1'] = data_testing[i+4*n_testing]['xss'][:,j]
            frame['sem2'] = data_testing[i+5*n_testing]['xss'][:,j]

            seq.append(frame)

        testing.append(seq)


    return training,validation,testing







