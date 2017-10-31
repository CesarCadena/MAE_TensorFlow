import scipy.io as io
import scipy.misc as misc
import os
import numpy as np
import matplotlib.pyplot as plt

from copy import copy


def load_data():
    with open('../../MAE_KITTI/kitti_split.txt') as file:
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

    basedir = '../../MAE_KITTI/data_18x60/'
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
    frame =  {'xcrLeft': 0,
              'xcgLeft': 0,
              'xcbLeft': 0,
              'xcrRight': 0,
              'xcgRight': 0,
              'xcbRight': 0,
              'xidsparseLeft': 0,
              'xidLeft': 0,
              'xmaskLeft': 0,
              'xidsparseRight': 0,
              'xidRight': 0,
              'xmaskRight': 0,
              'semLeft':0,
              'semRight':0}

    for i in range(0,n_training):
        seq = []
        for j in range(0,data_training[i]['xcr'].shape[1]):
            frame['xcrLeft'] = copy(data_training[i]['xcr'][:,j])
            frame['xcgLeft'] = copy(data_training[i]['xcg'][:,j])
            frame['xcbLeft'] = copy(data_training[i]['xcb'][:,j])

            frame['xcrRight'] = copy(data_training[i+n_training]['xcr'][:,j])
            frame['xcgRight'] = copy(data_training[i+n_training]['xcg'][:,j])
            frame['xcbRight'] = copy(data_training[i+n_training]['xcb'][:,j])

            frame['xidsparseLeft'] = copy(data_training[i+2*n_training]['xidsparse'][:,j])
            frame['xidLeft'] = copy(data_training[i+2*n_training]['xid'][:,j])
            frame['xmaskLeft'] = copy(data_training[i+2*n_training]['xmask'][:,j])

            frame['xidsparseRight'] = copy(data_training[i+3*n_training]['xidsparse'][:,j])
            frame['xidRight'] = copy(data_training[i+3*n_training]['xid'][:,j])
            frame['xmaskRight'] = copy(data_training[i+3*n_training]['xmask'][:,j])

            frame['semLeft'] = copy(data_training[i+4*n_training]['xss'][:,j])
            frame['semRight'] = copy(data_training[i+5*n_training]['xss'][:,j])

            frame1 = copy(frame)
            seq.append(frame1)
        seq1 = copy(seq)
        training.append(seq1)



    validation = []
    n_validation = len(data_val)

    for i in range(0,n_validation):
        seq = []

        for j in range(0,data_validation[i]['xcr'].shape[1]):
            frame['xcrLeft'] = copy(data_validation[i]['xcr'][:,j])
            frame['xcgLeft'] = copy(data_validation[i]['xcg'][:,j])
            frame['xcbLeft'] = copy(data_validation[i]['xcb'][:,j])

            frame['xcrRight'] = copy(data_validation[i+n_validation]['xcr'][:,j])
            frame['xcgRight'] = copy(data_validation[i+n_validation]['xcg'][:,j])
            frame['xcbRight'] = copy(data_validation[i+n_validation]['xcb'][:,j])

            frame['xidsparseLeft'] = copy(data_validation[i+2*n_validation]['xidsparse'][:,j])
            frame['xidLeft'] = copy(data_validation[i+2*n_validation]['xid'][:,j])
            frame['xmaskLeft'] = copy(data_validation[i+2*n_validation]['xmask'][:,j])

            frame['xidsparseRight'] = copy(data_validation[i+3*n_validation]['xidsparse'][:,j])
            frame['xidRight'] = copy(data_validation[i+3*n_validation]['xid'][:,j])
            frame['xmaskRight'] = copy(data_validation[i+3*n_validation]['xmask'][:,j])

            frame['semLeft'] = copy(data_validation[i+4*n_validation]['xss'][:,j])
            frame['semRight'] = copy(data_validation[i+5*n_validation]['xss'][:,j])
            frame1 = copy(frame)
            seq.append(frame1)

        seq1 = copy(seq)
        validation.append(seq1)


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
            frame['xcrLeft'] = copy(data_testing[i]['xcr'][:,j])
            frame['xcgLeft'] = copy(data_testing[i]['xcg'][:,j])
            frame['xcbLeft'] = copy(data_testing[i]['xcb'][:,j])

            frame['xcrRight'] = copy(data_testing[i+n_testing]['xcr'][:,j])
            frame['xcgRight'] = copy(data_testing[i+n_testing]['xcg'][:,j])
            frame['xcbRight'] = copy(data_testing[i+n_testing]['xcb'][:,j])

            frame['xidsparseLeft'] = copy(data_testing[i+2*n_testing]['xidsparse'][:,j])
            frame['xidLeft'] = copy(data_testing[i+2*n_testing]['xid'][:,j])
            frame['xmaskLeft'] = copy(data_testing[i+2*n_testing]['xmask'][:,j])

            frame['xidsparseRight'] = copy(data_testing[i+3*n_testing]['xidsparse'][:,j])
            frame['xidRight'] = copy(data_testing[i+3*n_testing]['xid'][:,j])
            frame['xmaskRight'] = copy(data_testing[i+3*n_testing]['xmask'][:,j])

            frame['semLeft'] = copy(data_testing[i+4*n_testing]['xss'][:,j])
            frame['semRight'] = copy(data_testing[i+5*n_testing]['xss'][:,j])

            frame1 = copy(frame)

            seq.append(frame1)

        seq1 = copy(seq)
        testing.append(seq1)

    return training,validation,testing

def load_split_data():
    with open('../../MAE_KITTI/kitti_split.txt') as file:
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

    return data_train, data_val, data_test

def load_set_data(data_seq):
        
    basedir = '../../MAE_KITTI/data_18x60/'
    basefile = 'data_kitti_'
    filetypes = ['im02','im03','InvDepth02','InvDepth03','seg02','seg03']

    data_set = []
    for i in filetypes:
        for j in data_seq:
            data_set.append(io.loadmat(basedir+basefile+i+'_'+j+'_18x60.mat'))

    # PREPARING TRAINING DATA

    width = 60
    height = 18

    im_shape = (width,height)

    n_seq = len(data_seq)

    dset = []
    frame =  {'xcrLeft': 0,
              'xcgLeft': 0,
              'xcbLeft': 0,
              'xcrRight': 0,
              'xcgRight': 0,
              'xcbRight': 0,
              'xidsparseLeft': 0,
              'xidLeft': 0,
              'xmaskLeft': 0,
              'xidsparseRight': 0,
              'xidRight': 0,
              'xmaskRight': 0,
              'semLeft':0,
              'semRight':0}

    for i in range(0,n_seq):
        seq = []        
        for j in range(0,data_set[i]['xcr'].shape[1]):
            frame['xcrLeft'] = copy(data_set[i]['xcr'][:,j])
            frame['xcgLeft'] = copy(data_set[i]['xcg'][:,j])
            frame['xcbLeft'] = copy(data_set[i]['xcb'][:,j])

            frame['xcrRight'] = copy(data_set[i+n_seq]['xcr'][:,j])
            frame['xcgRight'] = copy(data_set[i+n_seq]['xcg'][:,j])
            frame['xcbRight'] = copy(data_set[i+n_seq]['xcb'][:,j])

            frame['xidsparseLeft'] = copy(data_set[i+2*n_seq]['xidsparse'][:,j])
            frame['xidLeft'] = copy(data_set[i+2*n_seq]['xid'][:,j])
            frame['xmaskLeft'] = copy(data_set[i+2*n_seq]['xmask'][:,j])

            frame['xidsparseRight'] = copy(data_set[i+3*n_seq]['xidsparse'][:,j])
            frame['xidRight'] = copy(data_set[i+3*n_seq]['xid'][:,j])
            frame['xmaskRight'] = copy(data_set[i+3*n_seq]['xmask'][:,j])

            frame['semLeft'] = copy(data_set[i+4*n_seq]['xss'][:,j])
            frame['semRight'] = copy(data_set[i+5*n_seq]['xss'][:,j])

            frame1 = copy(frame)
            seq.append(frame1)
        seq1 = copy(seq)
        dset.append(seq1)
    
    return dset






