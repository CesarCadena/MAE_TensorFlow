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

    def __init__(self,state_size=None, coding_size=None, n_rnn_steps=None,scope=None, option='nonshared'):

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

        self.option = option

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

            if self.option == 'nonshared':

                for step in range(0,self.n_rnn_steps-1):

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

            if self.option == 'shared':

                self.H.append(tf.get_variable(name='H',
                                                  dtype=tf.float32,
                                                  initializer=self.initializer_H,
                                                  collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                               tf.GraphKeys.REGULARIZATION_LOSSES]))

                self.W.append(tf.get_variable(name='W',
                                              dtype=tf.float32,
                                              initializer=self.initializer_W,
                                              collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                           tf.GraphKeys.REGULARIZATION_LOSSES]))

                self.B.append(tf.get_variable(name='B',
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
        h_init = init_states

        def step(i,state,x):
            xi = tf.gather(x,i)
            if self.option == 'nonshared':
                state = tf.matmul(tf.add(tf.add(state,tf.matmul(xi,self.W[i])),self.B[i]),self.H[i])
            if self.option == 'shared':
                state = tf.matmul(tf.add(tf.add(state,tf.matmul(xi,self.W[0])),self.B[0]),self.H[0])

            state = tf.nn.relu(state)
            return i + 1, state, x


        _, h_fin, _ = tf.while_loop(lambda i, h, x: tf.less(i, self.n_rnn_steps),# this is called every iteration, if it outputs false, the loop stops
                                    step,
                                    [0,h_init,inputs])

        output = tf.matmul(tf.add(tf.add(tf.matmul(inputs[-1],self.W_t),h_fin),self.B_t),self.V_t)


        return tf.nn.relu(output)

class LSTM_RNN:

    def __init__(self, state_size=None, coding_size=None, n_rnn_steps=None, scope=None, option='shared'):

        # options
        if state_size == None:
            raise ValueError('no state size passed')
        self.size_states = state_size

        if coding_size == None:
            raise ValueError('no coding size passed')
        self.size_coding = coding_size

        if n_rnn_steps == None:
            raise ValueError('number of rnn steps not passed')
        self.n_rnn_steps = n_rnn_steps

        if scope == None:
            raise ValueError('no cell scope passed')
        self.scope = scope

        self.option = option

        # define initializer for input weights
        if state_size == coding_size:
            self.initializer_U = 0.0001*tf.diag(tf.ones([coding_size]))
        else:
            self.initializer_U = tf.concat([tf.diag(tf.ones([coding_size])), tf.zeros([coding_size, state_size - coding_size])], axis=1)

        # state-to-coding weights initializer
        if state_size == coding_size:
            self.initializer_O = tf.diag(tf.ones([coding_size]))
        else:
            self.initializer_O = tf.concat([tf.diag(tf.ones([coding_size])), tf.zeros([state_size - coding_size, coding_size])], axis=0)


        # forget gate
        # bias, input weights and recurrent weights for forget gate
        self.b_f = []
        self.U_f = []
        self.W_f = []

        # internal state
        # bias, input weights and recurrent weights for forget gate
        self.b = []
        self.U = []
        self.W = []

        # external input gate
        # with bias, input weights and recurrent weights for external input gate
        self.b_g = []
        self.U_g = []
        self.W_g = []

        # output gate
        # bias, input weights and recurrent weights for output gate
        self.b_o = []
        self.U_o = []
        self.W_o = []

        # define all variables
        with tf.variable_scope(scope) as lstm:




            if self.option == 'nonshared':
            # does not work with gradient clipping
                for step in range(0,self.n_rnn_steps):

                    # forget gate
                    self.b_f.append(tf.get_variable(name='b_f_'+str(step),
                                                    shape=[state_size],
                                                    dtype=tf.float32,
                                                    initializer=tf.zeros_initializer()))

                    self.U_f.append(tf.get_variable(name='U_f_'+str(step),
                                                    shape=[coding_size,state_size],
                                                    dtype=tf.float32,
                                                    initializer=tf.random_normal_initializer(mean=0.001,stddev=0.001)))

                    self.W_f.append(tf.get_variable(name='W_f_'+str(step),
                                                    shape=[state_size, state_size],
                                                    dtype=tf.float32,
                                                    initializer=tf.random_normal_initializer(mean=0.001,stddev=0.001)))


                    # internal state
                    self.b.append(tf.get_variable(name='b_'+str(step),
                                                  shape=[state_size],
                                                  dtype=tf.float32,
                                                  initializer=tf.zeros_initializer()))

                    self.U.append(tf.get_variable(name='U_'+str(step),
                                                  shape=[coding_size,state_size],
                                                  dtype=tf.float32,
                                                  initializer=tf.random_normal_initializer(mean=0.001,stddev=0.001)))

                    self.W.append(tf.get_variable(name='W_'+str(step),
                                                  shape=[state_size, state_size],
                                                  dtype=tf.float32,
                                                  initializer=tf.random_normal_initializer(mean=0.001,stddev=0.001)))

                    # external input gate
                    self.b_g.append(tf.get_variable(name='b_g_'+str(step),
                                                    shape=[state_size],
                                                    dtype=tf.float32,
                                                    initializer=tf.zeros_initializer()))

                    self.U_g.append(tf.get_variable(name='U_g_'+str(step),
                                                    shape=[coding_size,state_size],
                                                    dtype=tf.float32,
                                                    initializer=tf.random_normal_initializer(mean=0.1,stddev=0.1)))

                    self.W_g.append(tf.get_variable(name='W_g_'+str(step),
                                                    shape=[state_size, state_size],
                                                    dtype=tf.float32,
                                                    initializer=tf.random_normal_initializer(mean=0.0001,stddev=0.0001)))

                    # output gate
                    self.b_o.append(tf.get_variable(name='b_o_'+str(step),
                                                    shape=[state_size],
                                                    dtype=tf.float32,
                                                    initializer=tf.zeros_initializer()))

                    self.U_o.append(tf.get_variable(name='U_o_'+str(step),
                                                    shape=[coding_size,state_size],
                                                    dtype=tf.float32,
                                                    initializer=tf.random_normal_initializer(mean=0.1,stddev=0.1)))

                    self.W_o.append(tf.get_variable(name='W_o_'+str(step),
                                                    shape=[state_size, state_size],
                                                    dtype=tf.float32,
                                                    initializer=tf.random_normal_initializer(mean=0.0001,stddev=0.0001)))

            if self.option == 'shared':
                # forget gate
                    self.b_f.append(tf.get_variable(name='b_f',
                                                    shape=[state_size],
                                                    dtype=tf.float32,
                                                    initializer=tf.zeros_initializer()))

                    self.U_f.append(tf.get_variable(name='U_f',
                                                    shape=[coding_size,state_size],
                                                    dtype=tf.float32,
                                                    initializer=tf.random_normal_initializer(mean=0.0,stddev=0.1)))

                    self.W_f.append(tf.get_variable(name='W_f',
                                                    shape=[state_size, state_size],
                                                    dtype=tf.float32,
                                                    initializer=tf.random_normal_initializer(mean=0.0,stddev=0.1)))


                    # internal state
                    self.b.append(tf.get_variable(name='b',
                                                  shape=[state_size],
                                                  dtype=tf.float32,
                                                  initializer=tf.zeros_initializer()))

                    self.U.append(tf.get_variable(name='U',
                                                  shape=[coding_size,state_size],
                                                  dtype=tf.float32,
                                                  initializer=tf.random_normal_initializer(mean=0.0,stddev=0.1)))

                    self.W.append(tf.get_variable(name='W',
                                                  shape=[state_size, state_size],
                                                  dtype=tf.float32,
                                                  initializer=tf.random_normal_initializer(mean=0.,stddev=0.1)))

                    # external input gate
                    self.b_g.append(tf.get_variable(name='b_g',
                                                    shape=[state_size],
                                                    dtype=tf.float32,
                                                    initializer=tf.zeros_initializer()))

                    self.U_g.append(tf.get_variable(name='U_g',
                                                    shape=[coding_size,state_size],
                                                    dtype=tf.float32,
                                                    initializer=tf.random_normal_initializer(mean=0.0,stddev=0.1)))

                    self.W_g.append(tf.get_variable(name='W_g',
                                                    shape=[state_size, state_size],
                                                    dtype=tf.float32,
                                                    initializer=tf.random_normal_initializer(mean=0.0,stddev=0.1)))

                    # output gate
                    self.b_o.append(tf.get_variable(name='b_o',
                                                    shape=[state_size],
                                                    dtype=tf.float32,
                                                    initializer=tf.zeros_initializer()))

                    self.U_o.append(tf.get_variable(name='U_o',
                                                    shape=[coding_size,state_size],
                                                    dtype=tf.float32,
                                                    initializer=tf.random_normal_initializer(mean=0.0,stddev=0.1)))

                    self.W_o.append(tf.get_variable(name='W_o',
                                                    shape=[state_size, state_size],
                                                    dtype=tf.float32,
                                                    initializer=tf.random_normal_initializer(mean=0.0,stddev=0.1)))



             # state-to-coding weights
            self.O_w = tf.get_variable(name='O_w',
                                       shape=[state_size,coding_size],
                                       dtype=tf.float32,
                                       initializer=tf.random_normal_initializer(mean=0.0,stddev=0.1))

            self.O_b = tf.get_variable(name='O_b',
                                       shape=[coding_size],
                                       dtype=tf.float32,
                                       initializer=tf.zeros_initializer())

    def run(self,inputs,init_states=None):

        if init_states == None:
            raise ValueError('no state initialization passed')

        # state initialization
        h_init = init_states
        s_init = init_states

        def step(i,s_t,h_t,x):
            xi = tf.gather(x,i)
            f_t = tf.sigmoid(tf.add(tf.add(self.b_f[0],tf.matmul(xi,self.U_f[0])),tf.matmul(h_t,self.W_f[0])))
            i_t = tf.sigmoid(tf.add(tf.add(self.b_g[0],tf.matmul(xi,self.U_g[0])),tf.matmul(h_t,self.W_g[0])))
            g_t = tf.tanh(tf.add(tf.add(self.b[0],tf.matmul(xi,self.U[0])),tf.matmul(h_t,self.W[0])))
            s_t = tf.add(tf.multiply(f_t,s_t),tf.multiply(i_t,g_t))
            o_t = tf.sigmoid(tf.add(tf.add(self.b_o[0],tf.matmul(xi,self.U_o[0])),tf.matmul(h_t,self.W_o[0])))
            h_t = tf.multiply(o_t,tf.tanh(s_t))

            return i+1,s_t,h_t,x

        _, s_fin, h_fin, _ = tf.while_loop(lambda i,s,h,x: tf.less(i,self.n_rnn_steps),
                                           step,
                                           [0,s_init,h_init,inputs])


        # reconstruct coding
        output = tf.nn.relu(tf.add(tf.matmul(h_fin,self.O_w),self.O_b))
        return output

class Gated_RNN:

    def __init__(self,state_size=None, coding_size=None, n_rnn_steps=None, scope=None, option='shared'):

        # options
        if state_size == None:
            raise ValueError('no state size passed')
        self.size_states = state_size

        if coding_size == None:
            raise ValueError('no coding size passed')
        self.size_coding = coding_size

        if n_rnn_steps == None:
            raise ValueError('number of rnn steps not passed')
        self.n_rnn_steps = n_rnn_steps

        if scope == None:
            raise ValueError('no cell scope passed')
        self.scope = scope

        self.option = option

        # weight containers

        self.W_z = []
        self.U_z = []
        self.b_z = []

        self.W_r = []
        self.U_r = []
        self.b_r = []

        self.W = []
        self.U = []
        self.b = []

        # initializers

        if state_size == coding_size:
            initializer_U = tf.diag(tf.ones([state_size]))
            initializer_O = tf.diag(tf.ones([coding_size]))
        else:
            initializer_U = tf.concat(tf.diag(tf.ones([coding_size])),tf.zeros([coding_size,state_size-coding_size]),axis=1)
            initializer_O = tf.concat(tf.diag(tf.ones([coding_size])),tf.zeros([state_size-coding_size,coding_size]),axis=0)

        # define all variables
        with tf.variable_scope(scope) as gated:

            if self.option == 'shared':
                self.W_z.append(tf.get_variable(name='W_z',
                                                shape=[self.size_states,self.size_states],
                                                dtype=tf.float32,
                                                initializer=tf.random_normal_initializer(mean=0.0001,stddev=0.0001)))

                self.U_z.append(tf.get_variable(name='U_z',
                                                shape=[coding_size,state_size],
                                                dtype=tf.float32,
                                                initializer=tf.random_normal_initializer(mean=0.01,stddev=0.001)))

                self.b_z.append(tf.get_variable(name='b_z',
                                                shape=[self.size_states],
                                                dtype=tf.float32,
                                                initializer=tf.zeros_initializer()))

                self.W_r.append(tf.get_variable(name='W_r',
                                                shape=[self.size_states,self.size_states],
                                                dtype=tf.float32,
                                                initializer=tf.random_normal_initializer(mean=0.0001,stddev=0.0001)))

                self.U_r.append(tf.get_variable(name='U_r',
                                                shape=[coding_size,state_size],
                                                dtype=tf.float32,
                                                initializer=tf.random_normal_initializer(mean=0.01,stddev=0.001)))

                self.b_r.append(tf.get_variable(name='b_r',
                                                shape=[self.size_states],
                                                dtype=tf.float32,
                                                initializer=tf.zeros_initializer()))

                self.W.append(tf.get_variable(name='W',
                                              shape=[self.size_states,self.size_states],
                                              dtype=tf.float32,
                                              initializer=tf.random_normal_initializer(mean=0.0001,stddev=0.0001)))

                self.U.append(tf.get_variable(name='U',
                                              shape=[coding_size,state_size],
                                              dtype=tf.float32,
                                              initializer=tf.random_normal_initializer(mean=0.01,stddev=0.01)))

                self.b.append(tf.get_variable(name='b',
                                              shape=[self.size_states],
                                              dtype=tf.float32,
                                              initializer=tf.zeros_initializer()))


            if self.option == 'nonshared':
                for step in range(0,self.n_rnn_steps):

                    self.W_z.append(tf.get_variable(name='W_z_'+str(step),
                                                    shape=[self.size_states,self.size_states],
                                                    dtype=tf.float32,
                                                    initializer=tf.zeros_initializer()))

                    self.U_z.append(tf.get_variable(name='U_z_'+str(step),
                                                    dtype=tf.float32,
                                                    initializer=initializer_U))

                    self.b_z.append(tf.get_variable(name='b_z_'+str(step),
                                                    shape=[self.size_states],
                                                    dtype=tf.float32,
                                                    initializer=tf.zeros_initializer()))

                    self.W_r.append(tf.get_variable(name='W_r_'+str(step),
                                                    shape=[self.size_states,self.size_states],
                                                    dtype=tf.float32,
                                                    initializer=tf.zeros_initializer()))

                    self.U_r.append(tf.get_variable(name='U_r_'+str(step),
                                                    dtype=tf.float32,
                                                    initializer=initializer_U))

                    self.b_r.append(tf.get_variable(name='b_r_'+str(step),
                                                    shape=[self.size_states],
                                                    dtype=tf.float32,
                                                    initializer=tf.zeros_initializer()))

                    self.W.append(tf.get_variable(name='W_'+str(step),
                                                  shape=[self.size_states,self.size_states],
                                                  dtype=tf.float32,
                                                  initializer=tf.zeros_initializer()))

                    self.U.append(tf.get_variable(name='U_'+str(step),
                                                  dtype=tf.float32,
                                                  initializer=initializer_U))

                    self.b.append(tf.get_variable(name='b_'+str(step),
                                                  shape=[self.size_states],
                                                  dtype=tf.float32,
                                                  initializer=tf.zeros_initializer()))

            self.W_o_t = tf.get_variable(name='W_o_t',
                                         shape=[state_size,coding_size],
                                         dtype=tf.float32,
                                         initializer=tf.random_normal_initializer(mean=0.01,stddev=0.01))

            self.b_o_t = tf.get_variable(name='W_b_t',
                                         shape=[coding_size],
                                         dtype=tf.float32,
                                         initializer=tf.zeros_initializer())

    def run(self,inputs,init_states=None):

        if init_states == None:
            raise ValueError('no state initialization passed')

        h_init = init_states

        if self.option == 'shared':

            def step(i,h_t,x):
                z_t = tf.sigmoid(self.b_z[0] + tf.matmul(h_t,self.W_z[0]) + tf.matmul(inputs[0],self.U_z[0]))
                r_t = tf.sigmoid(self.b_r[0] + tf.matmul(h_t,self.W_r[0]) + tf.matmul(inputs[0],self.U_r[0]))
                r_t_tilde = tf.multiply(r_t,h_t)
                h_t_tilde = tf.tanh(self.b[0] + tf.matmul(r_t_tilde,self.W[0]) + tf.matmul(inputs[0],self.U[0]))

                h_t = h_t - tf.multiply(z_t,h_t) + tf.multiply(z_t,h_t_tilde)
                return i+1,h_t,x



        _, h_fin, _ = tf.while_loop(lambda  i,h,x: tf.less(i,self.n_rnn_steps),
                                    step,
                                    [0,h_init,inputs])



        output = tf.add(tf.matmul(h_fin,self.W_o_t),self.b_o_t)

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

    shd = tf.concat([dpt_ec,imr_ec,img_ec,imb_ec,sem],axis=1)

    shd_ec = SHD_EC.run(shd)
    shd_dc = SHD_DC.run(shd_ec)

    dpt_dc,imr_dc,img_dc,imb_dc,sem_dc = tf.split(shd_dc,num_or_size_splits=5,axis=1)

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

def RNN_MAE(imr,img,imb,dpt,gnd,obj,bld,veg,sky,n_rnn_steps=None,init_states=None,option=None,sharing='nonshared'):

    if n_rnn_steps == None:
        raise ValueError('no number of rnn steps passed')
    if init_states == None:
        raise ValueError('no state initialization passed')
    if option == None:
        raise ValueError('no RNN option passed')

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

    if option == 'basic':
        RNN = Basic_RNN(state_size=1024,coding_size=1024,n_rnn_steps=n_rnn_steps,scope='RNN',option=sharing)
    if option == 'lstm':
        RNN = LSTM_RNN(state_size=1024, coding_size=1024, n_rnn_steps=n_rnn_steps, scope='RNN',option=sharing)
    if option == 'gated':
        RNN = Gated_RNN(state_size=1024,coding_size=1024,n_rnn_steps=n_rnn_steps,scope='RNN')

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
