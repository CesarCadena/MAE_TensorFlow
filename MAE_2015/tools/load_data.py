import scipy.io as io
import scipy.misc as misc
import os
import numpy as np
import matplotlib.pyplot as plt

from copy import copy


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

def load_test_frames():
    with open('../../MAE_KITTI/test_frames.txt') as file:
        datatext = file.readlines()

    data_test = []
    for i in datatext:
        line = i.replace('\n','')
        line = line.replace('\r','')
        data_test.append(line)

    return data_test

def load_frames_data(test_frames):
        
    basedir = '../../MAE_KITTI/data_18x60/'
    basefile = 'data_kitti_'
    filetypes = ['im','InvDepth','seg']

    all_frames = []
    frame =  {'xcr': 0,
              'xcg': 0,
              'xcb': 0,              
              'xidsparse': 0,
              'xid': 0,
              'xmask': 0,              
              'sem':0}
    
    for j in test_frames:         
        seq_num = int(j[22:32])
        for i in filetypes:
            data = io.loadmat(basedir+basefile+i+j[33:]+'_'+j[:21]+'_18x60.mat')                        
            
            if i == 'im':
                frame['xcr'] = data['xcr'][:,seq_num]
                frame['xcg'] = data['xcg'][:,seq_num]
                frame['xcb'] = data['xcb'][:,seq_num]
            if i == 'InvDepth':
                frame['xidsparse'] = data['xidsparse'][:,seq_num]
                frame['xid'] = data['xid'][:,seq_num]
                frame['xmask'] = data['xmask'][:,seq_num]
            if i == 'seg':
                frame['sem'] = data['xss'][:,seq_num]
        
        frame1 = copy(frame)
        all_frames.append(frame1)
                
    return all_frames

