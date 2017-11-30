import tensorflow as tf
import numpy as np


class Encoding:

    def __init__(self,activation=None,shape_input=None,shape_coding=None,name=None):

        if name == None:
            raise ValueError('no name passed')

        if shape_input == None:
            raise ValueError('no input shape passed')

        if shape_coding == None:
            raise ValueError('no coding shape passed')

        if activation == None:
            raise ValueError('no activation passed')

        self.activation = activation

        # random initializer definition
        initializer = tf.random_normal([shape_input,shape_coding],stddev=0.01)

        # variable definition
        with tf.variable_scope("Encoding"):

            self.weight = tf.get_variable(name=name+'_ec_layer_weights',
                                          dtype=tf.float32,
                                          initializer=initializer,
                                          collections=[tf.GraphKeys.REGULARIZATION_LOSSES,
                                                       tf.GraphKeys.GLOBAL_VARIABLES])

            self.bias = tf.get_variable(name=name+'_ec_layer_bias',
                                        shape=[shape_coding],
                                        dtype=tf.float32,
                                        initializer=tf.zeros_initializer(),
                                        collections=[tf.GraphKeys.REGULARIZATION_LOSSES,
                                                     tf.GraphKeys.GLOBAL_VARIABLES])

    def run(self,x):

        # network definition
        z = tf.add(tf.matmul(x,self.weight),self.bias)

        # apply nonlinearity
        if self.activation == 'relu':
            output = tf.nn.relu(z)
        if self.activation == 'sigmoid':
            output = tf.nn.sigmoid(z)
        if self.activation == 'tanh':
            output = tf.nn.tanh(z)

        # return encoding
        return output

class Decoding:

    def __init__(self,activation=None,shape_coding=None,shape_decoding=None,name=None):

        if name == None:
            raise ValueError('no name passed')

        if shape_decoding == None:
            raise ValueError('no decoding shape passed')

        if shape_coding == None:
            raise ValueError('no coding shape passed')

        if activation == None:
            raise ValueError('no activation passed')

        self.activation = activation

        # random initializer definition
        initializer = tf.random_normal([shape_coding,shape_decoding],stddev=0.01)

        # variable definition
        with tf.variable_scope("Decoding"):

            self.weight = tf.get_variable(name=name+'_dc_layer_weights',
                                          dtype=tf.float32,
                                          initializer=initializer,
                                          collections=[tf.GraphKeys.REGULARIZATION_LOSSES,
                                                       tf.GraphKeys.GLOBAL_VARIABLES])

            self.bias = tf.get_variable(name=name+'_dc_layer_bias',
                                        shape=[shape_decoding],
                                        dtype=tf.float32,
                                        initializer=tf.zeros_initializer(),
                                        collections=[tf.GraphKeys.REGULARIZATION_LOSSES,
                                                     tf.GraphKeys.GLOBAL_VARIABLES])

    def run(self,x):

        # network definition
        z = tf.add(tf.matmul(x,self.weight),self.bias)

        # apply nonlinearity
        if self.activation == 'relu':
            output = tf.nn.relu(z)
        if self.activation == 'sigmoid':
            output = tf.nn.sigmoid(z)
        if self.activation == 'tanh':
            output = tf.nn.tanh(z)

        # return encoding
        return output

class Basic_RNN:

    def __init__(self,state_size=None, coding_size=None, n_rnn_steps=None,scope=None):

        # check if all parameters are well defined
        if state_size == None:
            raise ValueError('no state size passed')
        self.state_size = state_size

        if n_rnn_steps == None:
            raise ValueError('no number of rnn steps passed')
        self.n_rnn_steps = n_rnn_steps

        if coding_size == None:
            raise ValueError('no coding size passed')
        self.coding_size = coding_size

        if scope == None:
            raise ValueError('no scope name passed')
        self.scope = scope

        # container for recurrent weights
        self.H = []
        self.W = []
        self.B = []

        # initializer definition
        self.initializer_W = tf.concat([tf.diag(tf.ones([self.coding_size])),tf.zeros([self.coding_size,self.state_size-self.coding_size])],axis=1)
        self.initializer_H = 10e-5*tf.diag(tf.ones([state_size]))
        self.initializer_V = tf.concat([tf.diag(tf.ones([self.coding_size])),tf.zeros([self.state_size-self.coding_size,self.coding_size])],axis=0)

        # definition of rnn variables
        with tf.variable_scope(scope) as rnn:

            for step in range(0,self.n_rnn_steps):

                self.H.append(tf.get_variable(name='H_'+str(step),
                                              dtype=tf.float32,
                                              initializer=self.initializer_H,
                                              collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                           tf.GraphKeys.REGULARIZATION_LOSSES]))

                self.W.append(tf.get_variable(name='W'+str(step),
                                              dtype=tf.float32,
                                              initializer=self.initializer_W,
                                              collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                           tf.GraphKeys.REGULARIZATION_LOSSES]))

                self.B.append(tf.get_variable(name='B'+str(step),
                                              shape=[state_size],
                                              dtype=tf.float32,
                                              initializer=tf.zeros_initializer(),
                                              collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                           tf.GraphKeys.REGULARIZATION_LOSSES]))


            self.W_t = tf.get_variable(name='W_t',
                                       dtype=tf.float32,
                                       initializer=self.initializer_W,
                                       collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                    tf.GraphKeys.REGULARIZATION_LOSSES])

            self.B_t = tf.get_variable(name='B_t',
                                       shape=[state_size],
                                       dtype=tf.float32,
                                       initializer=tf.zeros_initializer(),
                                       collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                    tf.GraphKeys.REGULARIZATION_LOSSES])

            self.V_t = tf.get_variable(name='V_t',
                                       dtype=tf.float32,
                                       initializer=self.initializer_V,
                                       collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                    tf.GraphKeys.REGULARIZATION_LOSSES])

    def run(self,inputs,init_states=None):

        if init_states == None:
            raise ValueError('no state initialization passed')

        # initialize state of recurrent network from initializing placeholder
        state = init_states

        # running recurrent layer
        for i in range(0,self.n_rnn_steps):
            state = tf.matmul(tf.add(tf.add(state,tf.matmul(inputs[i],self.W[i])),self.B[i]),self.H[i])
            state = tf.nn.relu(state)

        output = tf.matmul(tf.add(tf.add(tf.matmul(inputs[-1],self.W_t),state),self.B_t),self.V_t)
        output = tf.nn.relu(output)

        return output

def LSTM_RNN(inputs,size_states=None,size_coding=None,n_rnn_steps=None, init_states=None, scope=None):
    # options
    if size_states == None:
        raise ValueError('no state size passed')

    if size_coding == None:
        raise ValueError('no coding size passed')

    if n_rnn_steps == None:
        raise ValueError('number of rnn steps not passed')

    if init_states == None:
        raise ValueError('no state initialization passed')

    if scope == None:
        raise ValueError('no cell scope passed')

    # define initializer for input weights
    if size_states == size_coding:
        initializer_U = 0.0001*tf.diag(tf.ones([size_coding]))
    else:
        initializer_U = tf.concat([tf.diag(tf.ones([size_coding])),tf.zeros([size_coding,size_states-size_coding])],axis=1)

    # state-to-coding weights initializer
    if size_states == size_coding:
        initializer_O = tf.diag(tf.ones([size_coding]))
    else:
        initializer_O = tf.concat([tf.diag(tf.ones([size_coding])),tf.zeros([size_states-size_coding,size_coding])],axis=0)

    # forget gate
    # bias, input weights and recurrent weights for forget gate
    b_f = []
    U_f = []
    W_f = []

    # internal state
    # bias, input weights and recurrent weights for forget gate
    b = []
    U = []
    W = []

    # external input gate
    # with bias, input weights and recurrent weights for external input gate
    b_g = []
    U_g = []
    W_g = []

    # output gate
    # bias, input weights and recurrent weights for output gate
    b_o = []
    U_o = []
    W_o = []

    # define all variables
    with tf.variable_scope(scope) as lstm:

        for step in range(0,n_rnn_steps):

            # forget gate
            b_f.append(tf.get_variable(name='b_f_'+str(step),
                                       shape=[size_states],
                                       dtype=tf.float32,
                                       initializer=tf.zeros_initializer()))

            U_f.append(tf.get_variable(name='U_f_'+str(step),
                                       dtype=tf.float32,
                                       initializer=initializer_U))

            W_f.append(tf.get_variable(name='W_f_'+str(step),
                                       shape=[size_states,size_states],
                                       dtype=tf.float32,
                                       initializer=tf.zeros_initializer()))


            # internal state
            b.append(tf.get_variable(name='b_'+str(step),
                                     shape=[size_states],
                                     dtype=tf.float32,
                                     initializer=tf.zeros_initializer()))

            U.append(tf.get_variable(name='U_'+str(step),
                                     dtype=tf.float32,
                                     initializer=initializer_U))

            W.append(tf.get_variable(name='W_'+str(step),
                                     shape=[size_states,size_states],
                                     dtype=tf.float32,
                                     initializer=tf.zeros_initializer()))

            # external input gate
            b_g.append(tf.get_variable(name='b_g_'+str(step),
                                       shape=[size_states],
                                       dtype=tf.float32,
                                       initializer=tf.zeros_initializer()))

            U_g.append(tf.get_variable(name='U_g_'+str(step),
                                       dtype=tf.float32,
                                       initializer=initializer_U))

            W_g.append(tf.get_variable(name='W_g_'+str(step),
                                       shape=[size_states,size_states],
                                       dtype=tf.float32,
                                       initializer=tf.zeros_initializer()))

            # output gate
            b_o.append(tf.get_variable(name='b_o_'+str(step),
                                       shape=[size_states],
                                       dtype=tf.float32,
                                       initializer=tf.zeros_initializer()))

            U_o.append(tf.get_variable(name='U_o_'+str(step),
                                       dtype=tf.float32,
                                       initializer=initializer_U))

            W_o.append(tf.get_variable(name='W_o_'+str(step),
                                       shape=[size_states,size_states],
                                       dtype=tf.float32,
                                       initializer=tf.zeros_initializer()))

             # state-to-coding weights
            O_w = tf.get_variable(name='O_w',
                                  dtype=tf.float32,
                                  initializer=initializer_O)

            O_b = tf.get_variable(name='O_b',
                                  shape=[size_coding],
                                  dtype=tf.float32,
                                  initializer=tf.zeros_initializer())

        rnn_variables = [v for v in tf.global_variables() if v.name.startswith(lstm.name)]



    # state initialization
    h_t = init_states
    s_t = init_states

    # cell definition
    for step in range(0,n_rnn_steps):

        # forget gate
        f_t = tf.sigmoid(b_f[step] + tf.matmul(inputs[step],U_f[step]) + tf.matmul(h_t,W_f[step]))

        # internal states
        i_t = tf.sigmoid(b_g[step] + tf.matmul(inputs[step],U_g[step]) + tf.matmul(h_t,W_g[step]))

        # external intput gate
        g_t = tf.tanh(b[step] + tf.matmul(inputs[step],U[step]) + tf.matmul(h_t,W[step]))

        # memory update
        s_t = tf.multiply(f_t,s_t) + tf.multiply(i_t,g_t)

        # output gate
        o_t = tf.sigmoid(b_o[step] + tf.matmul(inputs[step],U_o[step]) + tf.matmul(h_t,W_o[step]))

        # state update
        h_t = tf.multiply(o_t,tf.tanh(s_t))


    # reconstruct coding
    output = tf.add(tf.matmul(h_t,O_w),O_b)

    return output


def full_MAE(imr,img,imb,dpt,gnd,obj,bld,veg,sky):

    # initialize model
    IMR_EC = Encoding(activation='relu',shape_input=1080,shape_coding=1024,name='red')
    IMG_EC = Encoding(activation='relu',shape_input=1080,shape_coding=1024,name='green')
    IMB_EC = Encoding(activation='relu',shape_input=1080,shape_coding=1024,name='blue')
    DPT_EC = Encoding(activation='relu',shape_input=1080,shape_coding=1024,name='depth')
    GND_EC = Encoding(activation='relu',shape_input=1080,shape_coding=1024,name='gnd')
    OBJ_EC = Encoding(activation='relu',shape_input=1080,shape_coding=1024,name='obj')
    BLD_EC = Encoding(activation='relu',shape_input=1080,shape_coding=1024,name='bld')
    VEG_EC = Encoding(activation='relu',shape_input=1080,shape_coding=1024,name='veg')
    SKY_EC = Encoding(activation='relu',shape_input=1080,shape_coding=1024,name='sky')

    SEM_EC = Encoding(activation='relu',shape_input=5*1024,shape_coding=1024,name='sem')

    SHD_EC = Encoding(activation='relu',shape_input=5*1024,shape_coding=1024,name='full')
    SHD_DC = Decoding(activation='relu',shape_coding=1024,shape_decoding=5*1024,name='full')

    SEM_DC = Decoding(activation='relu',shape_coding=1024,shape_decoding=5*1024,name='sem')

    IMR_DC = Decoding(activation='sigmoid',shape_coding=1024,shape_decoding=1080,name='red')
    IMG_DC = Decoding(activation='sigmoid',shape_coding=1024,shape_decoding=1080,name='green')
    IMB_DC = Decoding(activation='sigmoid',shape_coding=1024,shape_decoding=1080,name='blue')
    DPT_DC = Decoding(activation='relu',shape_coding=1024,shape_decoding=1080,name='depth')
    GND_DC = Decoding(activation='sigmoid',shape_coding=1024,shape_decoding=1080,name='gnd')
    OBJ_DC = Decoding(activation='sigmoid',shape_coding=1024,shape_decoding=1080,name='obj')
    BLD_DC = Decoding(activation='sigmoid',shape_coding=1024,shape_decoding=1080,name='bld')
    VEG_DC = Decoding(activation='sigmoid',shape_coding=1024,shape_decoding=1080,name='veg')
    SKY_DC = Decoding(activation='sigmoid',shape_coding=1024,shape_decoding=1080,name='sky')


    # run model

    imr_ec = IMR_EC.run(imr)
    img_ec = IMG_EC.run(img)
    imb_ec = IMB_EC.run(imb)
    dpt_ec = DPT_EC.run(dpt)
    gnd_ec = GND_EC.run(gnd)
    obj_ec = OBJ_EC.run(obj)
    bld_ec = BLD_EC.run(bld)
    veg_ec = VEG_EC.run(veg)
    sky_ec = SKY_EC.run(sky)

    sem = tf.concat([gnd_ec,obj_ec,bld_ec,veg_ec,sky_ec],axis=1)
    sem = SEM_EC.run(sem)

    shd = tf.concat([imr_ec,img_ec,imb_ec,dpt_ec,sem],axis=1)

    shd_ec = SHD_EC.run(shd)
    shd_dc = SHD_DC.run(shd_ec)

    imr_dc,img_dc,imb_dc,dpt_dc,sem_dc = tf.split(shd_dc,num_or_size_splits=5,axis=1)

    sem_dc = SEM_DC.run(sem_dc)

    gnd_dc,obj_dc,bld_dc,veg_dc,sky_dc = tf.split(sem_dc,num_or_size_splits=5,axis=1)

    # decode all modalities

    imr_hat = IMR_DC.run(imr_dc)
    img_hat = IMG_DC.run(img_dc)
    imb_hat = IMB_DC.run(imb_dc)
    dpt_hat = DPT_DC.run(dpt_dc)
    gnd_hat = GND_DC.run(gnd_dc)
    obj_hat = OBJ_DC.run(obj_dc)
    bld_hat = BLD_DC.run(bld_dc)
    veg_hat = VEG_DC.run(veg_dc)
    sky_hat = SKY_DC.run(sky_dc)

    # generate output list
    output = [imr_hat,img_hat,imb_hat,dpt_hat,gnd_hat,obj_hat,bld_hat,veg_hat,sky_hat]
    return output


def RNN_MAE(imr,img,imb,dpt,gnd,obj,bld,veg,sky,n_rnn_steps=None,init_states=None):

    if n_rnn_steps == None:
        raise ValueError('no number of rnn steps passed')
    if init_states == None:
        raise ValueError('no state initialization passed')

    # initialize model
    IMR_EC = Encoding(activation='relu',shape_input=1080,shape_coding=1024,name='red')
    IMG_EC = Encoding(activation='relu',shape_input=1080,shape_coding=1024,name='green')
    IMB_EC = Encoding(activation='relu',shape_input=1080,shape_coding=1024,name='blue')
    DPT_EC = Encoding(activation='relu',shape_input=1080,shape_coding=1024,name='depth')
    GND_EC = Encoding(activation='relu',shape_input=1080,shape_coding=1024,name='gnd')
    OBJ_EC = Encoding(activation='relu',shape_input=1080,shape_coding=1024,name='obj')
    BLD_EC = Encoding(activation='relu',shape_input=1080,shape_coding=1024,name='bld')
    VEG_EC = Encoding(activation='relu',shape_input=1080,shape_coding=1024,name='veg')
    SKY_EC = Encoding(activation='relu',shape_input=1080,shape_coding=1024,name='sky')

    SEM_EC = Encoding(activation='relu',shape_input=5*1024,shape_coding=1024,name='sem')

    SHD_EC = Encoding(activation='relu',shape_input=5*1024,shape_coding=1024,name='full')

    RNN = Basic_RNN(state_size=1024,coding_size=1024,n_rnn_steps=n_rnn_steps,scope='RNN')

    SHD_DC = Decoding(activation='relu',shape_coding=1024,shape_decoding=5*1024,name='full')

    SEM_DC = Decoding(activation='relu',shape_coding=1024,shape_decoding=5*1024,name='sem')

    IMR_DC = Decoding(activation='sigmoid',shape_coding=1024,shape_decoding=1080,name='red')
    IMG_DC = Decoding(activation='sigmoid',shape_coding=1024,shape_decoding=1080,name='green')
    IMB_DC = Decoding(activation='sigmoid',shape_coding=1024,shape_decoding=1080,name='blue')
    DPT_DC = Decoding(activation='relu',shape_coding=1024,shape_decoding=1080,name='depth')
    GND_DC = Decoding(activation='sigmoid',shape_coding=1024,shape_decoding=1080,name='gnd')
    OBJ_DC = Decoding(activation='sigmoid',shape_coding=1024,shape_decoding=1080,name='obj')
    BLD_DC = Decoding(activation='sigmoid',shape_coding=1024,shape_decoding=1080,name='bld')
    VEG_DC = Decoding(activation='sigmoid',shape_coding=1024,shape_decoding=1080,name='veg')
    SKY_DC = Decoding(activation='sigmoid',shape_coding=1024,shape_decoding=1080,name='sky')

    # split the input placeholder according to the length of the sequence

    imr = tf.split(imr,n_rnn_steps,axis=1)
    img = tf.split(img,n_rnn_steps,axis=1)
    imb = tf.split(imb,n_rnn_steps,axis=1)
    dpt = tf.split(dpt,n_rnn_steps,axis=1)
    gnd = tf.split(gnd,n_rnn_steps,axis=1)
    obj = tf.split(obj,n_rnn_steps,axis=1)
    bld = tf.split(bld,n_rnn_steps,axis=1)
    veg = tf.split(veg,n_rnn_steps,axis=1)
    sky = tf.split(sky,n_rnn_steps,axis=1)

    # output container definition
    coding = []

    for step in range(0,n_rnn_steps):

        imr_in = tf.squeeze(imr[step],axis=1)
        img_in = tf.squeeze(img[step],axis=1)
        imb_in = tf.squeeze(imb[step],axis=1)
        dpt_in = tf.squeeze(dpt[step],axis=1)
        gnd_in = tf.squeeze(gnd[step],axis=1)
        obj_in = tf.squeeze(obj[step],axis=1)
        bld_in = tf.squeeze(bld[step],axis=1)
        veg_in = tf.squeeze(veg[step],axis=1)
        sky_in = tf.squeeze(sky[step],axis=1)

        imr_ec = IMR_EC.run(imr_in)
        img_ec = IMG_EC.run(img_in)
        imb_ec = IMB_EC.run(imb_in)
        dpt_ec = DPT_EC.run(dpt_in)
        gnd_ec = GND_EC.run(gnd_in)
        obj_ec = OBJ_EC.run(obj_in)
        bld_ec = BLD_EC.run(bld_in)
        veg_ec = VEG_EC.run(veg_in)
        sky_ec = SKY_EC.run(sky_in)

        sem = tf.concat([gnd_ec,obj_ec,bld_ec,veg_ec,sky_ec],axis=1)

        sem_ec = SEM_EC.run(sem)

        shd = tf.concat([dpt_ec,imr_ec,img_ec,imb_ec,sem_ec],axis=1)

        shd_ec = SHD_EC.run(shd)

        coding.append(shd_ec)

    rnn_output = RNN.run(coding,init_states=init_states)

    shd_dc = SHD_DC.run(rnn_output)

    dpt_dc,imr_dc,img_dc,imb_dc,sem_dc = tf.split(shd_dc,5,axis=1)

    sem_dc = SEM_DC.run(sem_dc)

    gnd_dc,obj_dc,bld_dc,veg_dc,sky_dc = tf.split(sem_dc,5,axis=1)

    # decode all modalities
    imr_hat = IMR_DC.run(imr_dc)
    img_hat = IMG_DC.run(img_dc)
    imb_hat = IMB_DC.run(imb_dc)
    dpt_hat = DPT_DC.run(dpt_dc)
    gnd_hat = GND_DC.run(gnd_dc)
    obj_hat = OBJ_DC.run(obj_dc)
    bld_hat = BLD_DC.run(bld_dc)
    veg_hat = VEG_DC.run(veg_dc)
    sky_hat = SKY_DC.run(sky_dc)

    # generate output list
    output = [imr_hat,img_hat,imb_hat,dpt_hat,gnd_hat,obj_hat,bld_hat,veg_hat,sky_hat]
    return output

















