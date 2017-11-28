import tensorflow as tf
import numpy as np


def encoding(x,
             activation=None,
             shape_input=None,
             shape_coding=None,
             name=None):

    if name == None:
        raise ValueError('no name passed')

    if shape_input == None:
        raise ValueError('no input shape passed')

    if shape_coding == None:
        raise ValueError('no coding shape passed')

    # random initializer definition
    initializer = tf.random_normal([shape_input,shape_coding],stddev=0.01)

    # variable definition
    with tf.variable_scope("Encoding") as scope:

        weight = tf.get_variable(name=name+'_ec_layer_weights',
                                 dtype=tf.float32,
                                 initializer=initializer)

        bias = tf.get_variable(name=name+'_ec_layer_bias',
                               shape=[shape_coding],
                               dtype=tf.float32,
                               initializer=tf.zeros_initializer())

    # network definition
    z = tf.add(tf.matmul(x,weight),bias)

    # apply nonlinearity
    if activation == None:
        output = tf.nn.relu(z)
    if activation == 'relu':
        output = tf.nn.relu(z)
    if activation == 'sigmoid':
        output = tf.nn.sigmoid(z)
    if activation == 'tanh':
        output = tf.nn.tanh(z)

    # return encoding
    return output

def decoding(x,
             activation=None,
             shape_coding = None,
             shape_decoding = None,
             name = None):

    if name == None:
        raise ValueError('no name passed')

    if shape_decoding == None:
        raise ValueError('no decoding shape passed')

    if shape_coding == None:
        raise ValueError('no coding shape passed')

    # random initializer definition
    initializer = tf.random_normal([shape_coding,shape_decoding],stddev=0.01)

    # variable definition
    with tf.variable_scope("Decoding") as scope:

        weight = tf.get_variable(name=name+'_dc_layer_weights',
                                 dtype=tf.float32,
                                 initializer=initializer)

        bias = tf.get_variable(name=name+'_dc_layer_bias',
                               shape=[shape_decoding],
                               dtype=tf.float32,
                               initializer=tf.zeros_initializer())

    # network definition
    z = tf.add(tf.matmul(x,weight),bias)

    # apply nonlinearity
    if activation == None:
        output = tf.nn.relu(z)
    if activation == 'relu':
        output = tf.nn.relu(z)
    if activation == 'sigmoid':
        output = tf.nn.sigmoid(z)
    if activation == 'tanh':
        output = tf.nn.tanh(z)

    # return encoding
    return output


def full_MAE(imr,img,imb,dpt,gnd,obj,bld,veg,sky):

    imr_ec = encoding(imr,activation='relu',shape_input=1080,shape_coding=1024,name='red')
    img_ec = encoding(img,activation='relu',shape_input=1080,shape_coding=1024,name='green')
    imb_ec = encoding(imb,activation='relu',shape_input=1080,shape_coding=1024,name='blue')
    dpt_ec = encoding(dpt,activation='relu',shape_input=1080,shape_coding=1024,name='depth')
    gnd_ec = encoding(gnd,activation='relu',shape_input=1080,shape_coding=1024,name='gnd')
    obj_ec = encoding(obj,activation='relu',shape_input=1080,shape_coding=1024,name='obj')
    bld_ec = encoding(bld,activation='relu',shape_input=1080,shape_coding=1024,name='bld')
    veg_ec = encoding(veg,activation='relu',shape_input=1080,shape_coding=1024,name='veg')
    sky_ec = encoding(sky,activation='relu',shape_input=1080,shape_coding=1024,name='sky')

    sem = tf.concat([gnd_ec,obj_ec,bld_ec,veg_ec,sky_ec],axis=1)

    sem_ec = encoding(sem,activation='relu',shape_input=5*1024,shape_coding=1024,name='sem')

    shd = tf.concat([imr_ec,img_ec,imb_ec,dpt_ec,sem_ec],axis=1)

    shd_ec = encoding(shd,activation='relu',shape_input=5*1024,shape_coding=1024,name='full')
    shd_dc = decoding(shd_ec,activation='relu',shape_coding=1024,shape_decoding=5*1024,name='full')

    imr_dc,img_dc,imb_dc,dpt_dc,sem_dc = tf.split(shd_dc,num_or_size_splits=5,axis=1)

    sem_dc = decoding(sem_dc,activation='relu',shape_coding=1024,shape_decoding=5*1024,name='sem')

    gnd_dc,obj_dc,bld_dc,veg_dc,sky_dc = tf.split(sem_dc,num_or_size_splits=5,axis=1)

    # decode all modalities
    imr_hat = decoding(imr_dc,activation='relu',shape_coding=1024,shape_decoding=1080,name='red')
    img_hat = decoding(img_dc,activation='relu',shape_coding=1024,shape_decoding=1080,name='green')
    imb_hat = decoding(imb_dc,activation='relu',shape_coding=1024,shape_decoding=1080,name='blue')
    dpt_hat = decoding(dpt_dc,activation='relu',shape_coding=1024,shape_decoding=1080,name='depth')
    gnd_hat = decoding(gnd_dc,activation='sigmoid',shape_coding=1024,shape_decoding=1080,name='gnd')
    obj_hat = decoding(obj_dc,activation='sigmoid',shape_coding=1024,shape_decoding=1080,name='obj')
    bld_hat = decoding(bld_dc,activation='sigmoid',shape_coding=1024,shape_decoding=1080,name='bld')
    veg_hat = decoding(veg_dc,activation='sigmoid',shape_coding=1024,shape_decoding=1080,name='veg')
    sky_hat = decoding(sky_dc,activation='sigmoid',shape_coding=1024,shape_decoding=1080,name='sky')

    # generate output list
    output = [imr_hat,img_hat,imb_hat,dpt_hat,gnd_hat,obj_hat,bld_hat,veg_hat,sky_hat]
    return output


















