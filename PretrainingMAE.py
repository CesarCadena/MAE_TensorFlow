
# coding: utf-8

# In[16]:

import tensorflow as tf
import numpy as np
from load_data import load_data
from input_distortion import input_distortion,pretraining_input_distortion
from visualization import print_training_frames,print_validation_frames
from copy import copy
import matplotlib.pyplot as plt

import datetime

data_train,data_validate,data_test = load_data()


class PretrainingMAE():

    def __init__(self,data_train,data_validate,data_test):

        # storing data
        self.data_train = data_train
        self.data_val = data_validate
        self.data_test = data_test

        # training options

        self.batch_size = 100
        self.hm_epochs = 150

        self.input_size = 1080
        self.hidden_size = 1024

        self.saving = True
        self.n_training_data = 'all'

        self.decay = 'constant'

        self.prepare_training_data()
        self.prepare_validation_data()

        self.n_validation_data = len(self.imr_val)
        self.n_training_validations = 100

        self.input_red =tf.placeholder('float', shape=[None, self.input_size])
        self.input_green = tf.placeholder('float',shape=[None,self.input_size])
        self.input_blue = tf.placeholder('float',shape=[None,self.input_size])
        self.input_depth = tf.placeholder('float',shape=[None,self.input_size])
        self.input_gnd = tf.placeholder('float',shape=[None,self.input_size])
        self.input_obj = tf.placeholder('float',shape=[None,self.input_size])
        self.input_bld = tf.placeholder('float',shape=[None,self.input_size])
        self.input_veg = tf.placeholder('float',shape=[None,self.input_size])
        self.input_sky = tf.placeholder('float',shape=[None,self.input_size])

        self.depth_mask = tf.placeholder('float',shape=[None,self.input_size])
        self.depth_loss_mask = tf.placeholder('float',shape=[None,self.input_size])

        self.label_red = tf.placeholder('float',shape=[None,self.input_size])
        self.label_green = tf.placeholder('float',shape=[None,self.input_size])
        self.label_blue = tf.placeholder('float',shape=[None,self.input_size])
        self.label_depth = tf.placeholder('float',shape=[None,self.input_size])
        self.label_gnd = tf.placeholder('float',shape=[None,self.input_size])
        self.label_obj = tf.placeholder('float',shape=[None,self.input_size])
        self.label_bld = tf.placeholder('float',shape=[None,self.input_size])
        self.label_veg = tf.placeholder('float',shape=[None,self.input_size])
        self.label_sky = tf.placeholder('float',shape=[None,self.input_size])

        # loss options


        # layers container

        self.layers = []

        self.model_folder = 'models'
        self.logs_folder = 'logs'

        # directory definitions
        self.project_dir ='./'
        self.model_dir = self.project_dir + self.model_folder
        self.logs_dir = self.project_dir + self.logs_folder


        tf.app.flags.DEFINE_string('train_dir',self.model_dir,'where to store the trained model')
        tf.app.flags.DEFINE_string('logs_dir',self.logs_dir,'where to store the summaries')

        self.FLAGS = tf.app.flags.FLAGS

    def prepare_training_data(self):
        '''
        Function for bringing the training data into a form that suits the training process
        :return:
        '''

        self.imr_train = []
        self.img_train = []
        self.imb_train = []
        self.depth_train = []
        self.gnd_train = []
        self.obj_train = []
        self.bld_train = []
        self.veg_train = []
        self.sky_train = []

        self.depth_mask_train = []
        self.depth_loss_mask_train = []

        t_iterator = 0

        for i in self.data_train:
            for j in i:
                if t_iterator == self.n_training_data:
                    break
                self.imr_train.append(j['xcr1']/255.)
                self.img_train.append(j['xcg1']/255.)
                self.imb_train.append(j['xcb1']/255.)
                self.depth_train.append(j['xid1'])
                self.depth_mask_train.append(j['xmask1'])
                self.depth_loss_mask_train.append((j['xid1']>0.09).astype(int))
                self.gnd_train.append((j['sem1']==1).astype(int))
                self.obj_train.append((j['sem1']==2).astype(int))
                self.bld_train.append((j['sem1']==3).astype(int))
                self.veg_train.append((j['sem1']==4).astype(int))
                self.sky_train.append((j['sem1']==5).astype(int))



                t_iterator += 1

        t_iterator = 0
        for i in self.data_train:
            for j in i:
                if t_iterator == self.n_training_data:
                    break
                self.imr_train.append(j['xcr2']/255.)
                self.img_train.append(j['xcg2']/255.)
                self.imb_train.append(j['xcb2']/255.)
                self.depth_train.append(j['xid2'])
                self.depth_mask_train.append(j['xmask2'])
                self.depth_loss_mask_train.append((j['xid2']>0.09).astype(int))
                self.gnd_train.append((j['sem2']==1).astype(int))
                self.obj_train.append((j['sem2']==2).astype(int))
                self.bld_train.append((j['sem2']==3).astype(int))
                self.veg_train.append((j['sem2']==4).astype(int))
                self.sky_train.append((j['sem2']==5).astype(int))

                t_iterator += 1

        # randomly shuffle input frames
        rand_indices = np.arange(len(self.imr_train)).astype(int)
        np.random.shuffle(rand_indices)

        self.imr_train = np.asarray(self.imr_train)[rand_indices]
        self.img_train = np.asarray(self.img_train)[rand_indices]
        self.imb_train = np.asarray(self.imb_train)[rand_indices]
        self.depth_train = np.asarray(self.depth_train)[rand_indices]
        self.gnd_train = np.asarray(self.gnd_train)[rand_indices]
        self.obj_train = np.asarray(self.obj_train)[rand_indices]
        self.bld_train = np.asarray(self.bld_train)[rand_indices]
        self.veg_train = np.asarray(self.veg_train)[rand_indices]
        self.sky_train = np.asarray(self.sky_train)[rand_indices]
        self.depth_mask_train = np.asarray(self.depth_mask_train)[rand_indices]
        self.depth_loss_mask_train = np.asarray(self.depth_loss_mask_train)[rand_indices]

    def prepare_validation_data(self):

            # prepare validation data containers
            self.imr_val = []
            self.img_val = []
            self.imb_val = []
            self.depth_val = []
            self.gnd_val = []
            self.obj_val = []
            self.bld_val = []
            self.veg_val = []
            self.sky_val = []
            self.depth_mask_val = []
            self.depth_loss_mask_val = []

            v_iterator = 0

            for i in self.data_val:
                for j in i:
                    self.imr_val.append(j['xcr1']/255.)
                    self.img_val.append(j['xcg1']/255.)
                    self.imb_val.append(j['xcb1']/255.)
                    self.depth_val.append(j['xid1'])
                    self.depth_loss_mask_val.append((j['xid1']>0.1).astype(int))
                    self.gnd_val.append((j['sem1']==1).astype(int))
                    self.obj_val.append((j['sem1']==2).astype(int))
                    self.bld_val.append((j['sem1']==3).astype(int))
                    self.veg_val.append((j['sem1']==4).astype(int))
                    self.sky_val.append((j['sem1']==5).astype(int))
                    self.depth_mask_val.append(j['xmask1'])

                    v_iterator += 1

    def AE_red(self,input):

        self.red_ec_layer = {'weights':tf.Variable(tf.random_normal(shape=[self.input_size, self.hidden_size], stddev=0.01), name='red_ec_layer_weights'),
                             'bias':tf.Variable(tf.zeros([self.hidden_size]), name="red_ec_layer_bias")}
        self.layers.append([self.red_ec_layer])

        hidden_red = tf.nn.relu(tf.add(tf.matmul(input, self.red_ec_layer['weights']),self.red_ec_layer['bias']))

        self.red_dc_layer = {'weights':tf.Variable(tf.random_normal(shape=[self.hidden_size, self.input_size], stddev=0.01), name='red_dc_layer_weights'),
                             'bias':tf.Variable(tf.zeros([self.input_size]), name="red_dc_layer_bias")}
        self.layers.append(self.red_dc_layer)

        output = tf.nn.sigmoid(tf.add(tf.matmul(hidden_red, self.red_dc_layer['weights']),self.red_dc_layer['bias']))

        return output

    def AE_green(self,input):

        self.green_ec_layer = {'weights':tf.Variable(tf.random_normal(shape=[self.input_size, self.hidden_size], stddev=0.01), name='green_ec_layer_weights'),
                             'bias':tf.Variable(tf.zeros([self.hidden_size]), name="green_ec_layer_bias")}
        self.layers.append([self.green_ec_layer])

        hidden_green = tf.nn.relu(tf.add(tf.matmul(input, self.green_ec_layer['weights']),self.green_ec_layer['bias']))

        self.green_dc_layer = {'weights':tf.Variable(tf.random_normal(shape=[self.hidden_size, self.input_size], stddev=0.01), name='green_dc_layer_weights'),
                             'bias':tf.Variable(tf.zeros([self.input_size]), name="green_dc_layer_bias")}
        self.layers.append(self.green_dc_layer)

        output = tf.nn.sigmoid(tf.add(tf.matmul(hidden_green, self.green_dc_layer['weights']),self.green_dc_layer['bias']))

        return output

    def AE_blue(self,input):

        self.blue_ec_layer = {'weights':tf.Variable(tf.random_normal(shape=[self.input_size, self.hidden_size], stddev=0.01), name='blue_ec_layer_weights'),
                             'bias':tf.Variable(tf.zeros([self.hidden_size]), name="blue_ec_layer_bias")}
        self.layers.append([self.blue_ec_layer])

        hidden_blue = tf.nn.relu(tf.add(tf.matmul(input, self.blue_ec_layer['weights']),self.blue_ec_layer['bias']))

        self.blue_dc_layer = {'weights':tf.Variable(tf.random_normal(shape=[self.hidden_size, self.input_size], stddev=0.01), name='blue_dc_layer_weights'),
                             'bias':tf.Variable(tf.zeros([self.input_size]), name="blue_dc_layer_bias")}
        self.layers.append(self.blue_dc_layer)

        output = tf.nn.sigmoid(tf.add(tf.matmul(hidden_blue, self.blue_dc_layer['weights']),self.blue_dc_layer['bias']))

        return output

    def AE_depth(self,input):

        self.depth_ec_layer = {'weights':tf.Variable(tf.random_normal(shape=[self.input_size, self.hidden_size], stddev=0.01), name='depth_ec_layer_weights'),
                             'bias':tf.Variable(tf.zeros([self.hidden_size]), name="depth_ec_layer_bias")}
        self.layers.append([self.depth_ec_layer])

        hidden_depth = tf.nn.relu(tf.add(tf.matmul(input, self.depth_ec_layer['weights']),self.depth_ec_layer['bias']))

        self.depth_dc_layer = {'weights':tf.Variable(tf.random_normal(shape=[self.hidden_size, self.input_size], stddev=0.01), name='depth_dc_layer_weights'),
                             'bias':tf.Variable(tf.zeros([self.input_size]), name="depth_dc_layer_bias")}
        self.layers.append(self.depth_dc_layer)

        output = tf.add(tf.matmul(hidden_depth, self.depth_dc_layer['weights']),self.depth_dc_layer['bias'])

        return output

    def AE_gnd(self,input):

        self.gnd_ec_layer = {'weights':tf.Variable(tf.random_normal(shape=[self.input_size, self.hidden_size], stddev=0.01), name='gnd_ec_layer_weights'),
                             'bias':tf.Variable(tf.zeros([self.hidden_size]), name="gnd_ec_layer_bias")}
        self.layers.append([self.gnd_ec_layer])

        hidden_gnd = tf.nn.relu(tf.add(tf.matmul(input, self.gnd_ec_layer['weights']),self.gnd_ec_layer['bias']))

        self.gnd_dc_layer = {'weights':tf.Variable(tf.random_normal(shape=[self.hidden_size, self.input_size], stddev=0.01), name='gnd_dc_layer_weights'),
                             'bias':tf.Variable(tf.zeros([self.input_size]), name="gnd_dc_layer_bias")}
        self.layers.append(self.gnd_dc_layer)

        output = tf.nn.sigmoid(tf.add(tf.matmul(hidden_gnd, self.gnd_dc_layer['weights']),self.gnd_dc_layer['bias']))

        return output

    def AE_obj(self,input):

        self.obj_ec_layer = {'weights':tf.Variable(tf.random_normal(shape=[self.input_size, self.hidden_size], stddev=0.01), name='obj_ec_layer_weights'),
                             'bias':tf.Variable(tf.zeros([self.hidden_size]), name="obj_ec_layer_bias")}
        self.layers.append([self.obj_ec_layer])

        hidden_obj = tf.nn.relu(tf.add(tf.matmul(input, self.obj_ec_layer['weights']),self.obj_ec_layer['bias']))

        self.obj_dc_layer = {'weights':tf.Variable(tf.random_normal(shape=[self.hidden_size, self.input_size], stddev=0.01), name='obj_dc_layer_weights'),
                             'bias':tf.Variable(tf.zeros([self.input_size]), name="obj_dc_layer_bias")}
        self.layers.append(self.obj_dc_layer)

        output = tf.nn.sigmoid(tf.add(tf.matmul(hidden_obj, self.obj_dc_layer['weights']),self.obj_dc_layer['bias']))

        return output

    def AE_bld(self,input):

        self.bld_ec_layer = {'weights':tf.Variable(tf.random_normal(shape=[self.input_size, self.hidden_size], stddev=0.01), name='bld_ec_layer_weights'),
                             'bias':tf.Variable(tf.zeros([self.hidden_size]), name="bld_ec_layer_bias")}
        self.layers.append([self.bld_ec_layer])

        hidden_bld = tf.nn.relu(tf.add(tf.matmul(input, self.bld_ec_layer['weights']),self.bld_ec_layer['bias']))

        self.bld_dc_layer = {'weights':tf.Variable(tf.random_normal(shape=[self.hidden_size, self.input_size], stddev=0.01), name='bld_dc_layer_weights'),
                             'bias':tf.Variable(tf.zeros([self.input_size]), name="bld_dc_layer_bias")}
        self.layers.append(self.bld_dc_layer)

        output = tf.nn.sigmoid(tf.add(tf.matmul(hidden_bld, self.bld_dc_layer['weights']),self.bld_dc_layer['bias']))

        return output

    def AE_veg(self,input):

        self.veg_ec_layer = {'weights':tf.Variable(tf.random_normal(shape=[self.input_size, self.hidden_size], stddev=0.01), name='veg_ec_layer_weights'),
                             'bias':tf.Variable(tf.zeros([self.hidden_size]), name="veg_ec_layer_bias")}
        self.layers.append([self.veg_ec_layer])

        hidden_veg = tf.nn.relu(tf.add(tf.matmul(input, self.veg_ec_layer['weights']),self.veg_ec_layer['bias']))

        self.veg_dc_layer = {'weights':tf.Variable(tf.random_normal(shape=[self.hidden_size, self.input_size], stddev=0.01), name='veg_dc_layer_weights'),
                             'bias':tf.Variable(tf.zeros([self.input_size]), name="veg_dc_layer_bias")}
        self.layers.append(self.veg_dc_layer)

        output = tf.nn.sigmoid(tf.add(tf.matmul(hidden_veg, self.veg_dc_layer['weights']),self.veg_dc_layer['bias']))

        return output

    def AE_sky(self,input):

        self.sky_ec_layer = {'weights':tf.Variable(tf.random_normal(shape=[self.input_size, self.hidden_size], stddev=0.01), name='sky_ec_layer_weights'),
                             'bias':tf.Variable(tf.zeros([self.hidden_size]), name="sky_ec_layer_bias")}
        self.layers.append([self.sky_ec_layer])

        hidden_sky = tf.nn.relu(tf.add(tf.matmul(input, self.sky_ec_layer['weights']),self.sky_ec_layer['bias']))

        self.sky_dc_layer = {'weights':tf.Variable(tf.random_normal(shape=[self.hidden_size, self.input_size], stddev=0.01), name='sky_dc_layer_weights'),
                             'bias':tf.Variable(tf.zeros([self.input_size]), name="sky_dc_layer_bias")}

        self.layers.append(self.sky_dc_layer)

        output = tf.nn.sigmoid(tf.add(tf.matmul(hidden_sky, self.sky_dc_layer['weights']),self.sky_dc_layer['bias']))

        return output

    def AE_sem(self,gnd,obj,bld,veg,sky):

        hidden_gnd = tf.nn.relu(tf.add(tf.matmul(gnd,self.gnd_ec_layer['weights']),self.gnd_ec_layer['bias']))
        hidden_obj = tf.nn.relu(tf.add(tf.matmul(obj,self.obj_ec_layer['weights']),self.obj_ec_layer['bias']))
        hidden_bld = tf.nn.relu(tf.add(tf.matmul(bld,self.bld_ec_layer['weights']),self.bld_ec_layer['bias']))
        hidden_veg = tf.nn.relu(tf.add(tf.matmul(veg,self.veg_ec_layer['weights']),self.veg_ec_layer['bias']))
        hidden_sky = tf.nn.relu(tf.add(tf.matmul(sky,self.sky_ec_layer['weights']),self.sky_ec_layer['bias']))

        sem = tf.concat([hidden_gnd,hidden_obj,hidden_bld,hidden_veg,hidden_sky],axis=1)

        self.sem_ec_layer = {'weights':tf.Variable(tf.random_normal(shape=[5*self.hidden_size, self.hidden_size], stddev=0.01), name='sem_ec_layer_weights'),
                             'bias':tf.Variable(tf.zeros([self.hidden_size]), name="sem_ec_layer_bias")}
        self.layers.append(self.sem_ec_layer)

        hidden_sem = tf.nn.relu(tf.add(tf.matmul(sem,self.sem_ec_layer['weights']),self.sem_ec_layer['bias']))

        self.sem_dc_layer = {'weights':tf.Variable(tf.random_normal(shape=[self.hidden_size, 5*self.hidden_size], stddev=0.01), name='sem_dc_layer_weights'),
                             'bias':tf.Variable(tf.zeros([5*self.hidden_size]), name="sem_dc_layer_bias")}
        self.layers.append(self.sem_dc_layer)

        sem_out = tf.nn.relu(tf.add(tf.matmul(hidden_sem,self.sem_dc_layer['weights']),self.sem_dc_layer['bias']))

        hidden_gnd_out,hidden_obj_out,hidden_bld_out,hidden_veg_out,hidden_sky_out = tf.split(sem_out,5,axis=1)


        gnd_output = tf.sigmoid(tf.add(tf.matmul(hidden_gnd_out,self.gnd_dc_layer['weights']),self.gnd_dc_layer['bias']))
        obj_output = tf.sigmoid(tf.add(tf.matmul(hidden_obj_out,self.obj_dc_layer['weights']),self.obj_dc_layer['bias']))
        bld_output = tf.sigmoid(tf.add(tf.matmul(hidden_bld_out,self.bld_dc_layer['weights']),self.bld_dc_layer['bias']))
        veg_output = tf.sigmoid(tf.add(tf.matmul(hidden_veg_out,self.veg_dc_layer['weights']),self.veg_dc_layer['bias']))
        sky_output = tf.sigmoid(tf.add(tf.matmul(hidden_sky_out,self.sky_dc_layer['weights']),self.sky_dc_layer['bias']))

        return gnd_output,obj_output,bld_output,veg_output,sky_output

    def pretrain_seperate_channels(self):

        red_pred = self.AE_red(self.input_red)
        green_pred = self.AE_green(self.input_green)
        blue_pred = self.AE_blue(self.input_blue)
        depth_pred = self.AE_depth(self.input_depth)
        gnd_pred = self.AE_gnd(self.input_gnd)
        obj_pred = self.AE_obj(self.input_obj)
        bld_pred = self.AE_bld(self.input_bld)
        veg_pred = self.AE_veg(self.input_veg)
        sky_pred = self.AE_sky(self.input_sky)

        cost_red = tf.nn.l2_loss(red_pred-self.label_red)
        cost_green = tf.nn.l2_loss(green_pred-self.label_green)
        cost_blue = tf.nn.l2_loss(blue_pred-self.label_blue)
        cost_depth = tf.nn.l2_loss(tf.multiply(self.depth_mask,depth_pred)-tf.multiply(self.depth_mask,self.label_depth))
        cost_gnd = tf.nn.l2_loss(gnd_pred-self.label_gnd)
        cost_obj = tf.nn.l2_loss(obj_pred-self.label_obj)
        cost_bld = tf.nn.l2_loss(bld_pred-self.label_bld)
        cost_veg = tf.nn.l2_loss(veg_pred-self.label_veg)
        cost_sky = tf.nn.l2_loss(sky_pred-self.label_sky)

        regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)

        reg_red = tf.contrib.layers.apply_regularization(regularizer,weights_list=[self.red_ec_layer['weights'],
                                                                             self.red_dc_layer['weights'],
                                                                             self.red_ec_layer['bias'],
                                                                             self.red_dc_layer['bias']])

        reg_green = tf.contrib.layers.apply_regularization(regularizer,weights_list=[self.green_ec_layer['weights'],
                                                                             self.green_dc_layer['weights'],
                                                                             self.green_ec_layer['bias'],
                                                                             self.green_dc_layer['bias']])

        reg_blue = tf.contrib.layers.apply_regularization(regularizer,weights_list=[self.blue_ec_layer['weights'],
                                                                             self.blue_dc_layer['weights'],
                                                                             self.blue_ec_layer['bias'],
                                                                             self.blue_dc_layer['bias']])

        reg_depth = tf.contrib.layers.apply_regularization(regularizer,weights_list=[self.depth_ec_layer['weights'],
                                                                             self.depth_dc_layer['weights'],
                                                                             self.depth_ec_layer['bias'],
                                                                             self.depth_dc_layer['bias']])

        reg_gnd = tf.contrib.layers.apply_regularization(regularizer,weights_list=[self.gnd_ec_layer['weights'],
                                                                             self.gnd_dc_layer['weights'],
                                                                             self.gnd_ec_layer['bias'],
                                                                             self.gnd_dc_layer['bias']])

        reg_obj = tf.contrib.layers.apply_regularization(regularizer,weights_list=[self.obj_ec_layer['weights'],
                                                                             self.obj_dc_layer['weights'],
                                                                             self.obj_ec_layer['bias'],
                                                                             self.obj_dc_layer['bias']])

        reg_bld = tf.contrib.layers.apply_regularization(regularizer,weights_list=[self.bld_ec_layer['weights'],
                                                                             self.bld_dc_layer['weights'],
                                                                             self.bld_ec_layer['bias'],
                                                                             self.bld_dc_layer['bias']])

        reg_veg = tf.contrib.layers.apply_regularization(regularizer,weights_list=[self.veg_ec_layer['weights'],
                                                                             self.veg_dc_layer['weights'],
                                                                             self.veg_ec_layer['bias'],
                                                                             self.veg_dc_layer['bias']])

        reg_sky = tf.contrib.layers.apply_regularization(regularizer,weights_list=[self.sky_ec_layer['weights'],
                                                                             self.sky_dc_layer['weights'],
                                                                             self.sky_ec_layer['bias'],
                                                                             self.sky_dc_layer['bias']])

        cost_red += reg_red
        cost_green += reg_green
        cost_blue += reg_blue
        cost_depth += reg_depth
        cost_gnd += reg_gnd
        cost_obj += reg_obj
        cost_bld += reg_bld
        cost_veg += reg_veg
        cost_sky += reg_sky


        summary_red = tf.summary.scalar('cost_red',cost_red)
        summary_green = tf.summary.scalar('cost_green',cost_green)
        summary_blue = tf.summary.scalar('cost_blue',cost_blue)
        summary_depth = tf.summary.scalar('cost_depth',cost_depth)
        summary_gnd = tf.summary.scalar('cost_gnd',cost_gnd)
        summary_obj = tf.summary.scalar('cost_obj',cost_obj)
        summary_bld = tf.summary.scalar('cost_bld',cost_bld)
        summary_veg = tf.summary.scalar('cost_veg',cost_veg)
        summary_sky = tf.summary.scalar('cost_sky',cost_sky)

        opt_red = tf.train.AdamOptimizer(learning_rate=0.000001).minimize(cost_red)
        opt_green = tf.train.AdamOptimizer(learning_rate=0.000001).minimize(cost_green)
        opt_blue = tf.train.AdamOptimizer(learning_rate=0.000001).minimize(cost_blue)
        opt_depth = tf.train.AdamOptimizer(learning_rate=0.000001).minimize(cost_depth)
        opt_gnd = tf.train.AdamOptimizer(learning_rate=0.000001).minimize(cost_gnd)
        opt_obj = tf.train.AdamOptimizer(learning_rate=0.000001).minimize(cost_obj)
        opt_bld = tf.train.AdamOptimizer(learning_rate=0.000001).minimize(cost_bld)
        opt_veg = tf.train.AdamOptimizer(learning_rate=0.000001).minimize(cost_veg)
        opt_sky = tf.train.AdamOptimizer(learning_rate=0.000001).minimize(cost_sky)



        with tf.Session() as sess:

            train_writer1 = tf.summary.FileWriter(self.FLAGS.logs_dir,sess.graph)
            sess.run(tf.global_variables_initializer())

            n_batches = int(len(self.imr_train)/self.batch_size)
            epoch_losses = []

            for epoch in range(0,self.hm_epochs):
                epoch_loss = np.zeros((9,1))
                for _ in range(0,n_batches):

                    imr_batch = self.imr_train[_*self.batch_size:(_+1)*self.batch_size]
                    img_batch = self.img_train[_*self.batch_size:(_+1)*self.batch_size]
                    imb_batch = self.imb_train[_*self.batch_size:(_+1)*self.batch_size]
                    depth_batch = self.depth_train[_*self.batch_size:(_+1)*self.batch_size]
                    gnd_batch = self.gnd_train[_*self.batch_size:(_+1)*self.batch_size]
                    obj_batch = self.obj_train[_*self.batch_size:(_+1)*self.batch_size]
                    bld_batch = self.bld_train[_*self.batch_size:(_+1)*self.batch_size]
                    veg_batch = self.veg_train[_*self.batch_size:(_+1)*self.batch_size]
                    sky_batch = self.sky_train[_*self.batch_size:(_+1)*self.batch_size]
                    depth_mask = self.depth_mask_train[_*self.batch_size:(_+1)*self.batch_size]

                    imr_in,img_in,imb_in,depth_in,gnd_in,obj_in,bld_in,veg_in,sky_in = pretraining_input_distortion(imr_batch,
                                                                                                                    img_batch,
                                                                                                                    imb_batch,
                                                                                                                    depth_batch,
                                                                                                                    gnd_batch,
                                                                                                                    obj_batch,
                                                                                                                    bld_batch,
                                                                                                                    veg_batch,
                                                                                                                    sky_batch,
                                                                                                                    resolution=(18,60))

                    feed_dict_red = {self.input_red:imr_in,
                                     self.label_red:imr_batch}

                    feed_dict_green = {self.input_green:img_in,
                                     self.label_green:img_batch}

                    feed_dict_blue = {self.input_blue:imb_in,
                                     self.label_blue:imb_batch}

                    feed_dict_depth = {self.input_depth:depth_in,
                                       self.depth_mask:depth_mask,
                                       self.label_depth:depth_batch}

                    feed_dict_gnd = {self.input_gnd:gnd_in,
                                     self.label_gnd:gnd_batch}
                    feed_dict_obj = {self.input_obj:obj_in,
                                     self.label_obj:obj_batch}
                    feed_dict_bld = {self.input_bld:bld_in,
                                     self.label_bld:bld_batch}
                    feed_dict_veg = {self.input_veg:veg_in,
                                     self.label_veg:veg_batch}
                    feed_dict_sky = {self.input_sky:sky_in,
                                     self.label_sky:sky_batch}

                    sum_red, _, c_red = sess.run([summary_red, opt_red, cost_red], feed_dict=feed_dict_red)
                    epoch_loss[0] += c_red

                    sum_green, _, c_green = sess.run([summary_green, opt_green, cost_green], feed_dict=feed_dict_green)
                    epoch_loss[1] += c_green

                    sum_blue, _, c_blue = sess.run([summary_blue, opt_blue, cost_blue],feed_dict=feed_dict_blue)
                    epoch_loss[2] += c_blue

                    sum_depth, _, c_depth = sess.run([summary_depth, opt_depth,cost_depth],feed_dict=feed_dict_depth)
                    epoch_loss[3] += c_depth

                    sum_gnd, _, c_gnd = sess.run([summary_gnd, opt_gnd, cost_gnd], feed_dict=feed_dict_gnd)
                    epoch_loss[4] += c_gnd

                    sum_obj, _, c_obj = sess.run([summary_obj, opt_obj, cost_obj], feed_dict=feed_dict_obj)
                    epoch_loss[5] += c_obj

                    sum_bld, _, c_bld = sess.run([summary_bld, opt_bld, cost_bld],feed_dict=feed_dict_bld)
                    epoch_loss[6] += c_bld

                    sum_veg, _, c_veg = sess.run([summary_veg, opt_veg, cost_veg], feed_dict=feed_dict_veg)
                    epoch_loss[7] += c_veg

                    sum_sky, _, c_sky = sess.run([summary_sky, opt_sky, cost_sky],feed_dict=feed_dict_sky)
                    epoch_loss[8] += c_sky

                epoch_losses.append(epoch_loss)

                train_writer1.add_summary(sum_red,epoch)
                train_writer1.add_summary(sum_green,epoch)
                train_writer1.add_summary(sum_blue,epoch)
                train_writer1.add_summary(sum_depth,epoch)
                train_writer1.add_summary(sum_gnd,epoch)
                train_writer1.add_summary(sum_obj,epoch)
                train_writer1.add_summary(sum_bld,epoch)
                train_writer1.add_summary(sum_veg,epoch)
                train_writer1.add_summary(sum_sky,epoch)

                print('Epoch', epoch, 'completed out of', self.hm_epochs)
                print('Loss Red Channel: ', epoch_losses[epoch][0])
                print('Loss Green Channel: ', epoch_losses[epoch][1])
                print('Loss Blue Channel: ', epoch_losses[epoch][2])
                print('Loss Depth Channel: ', epoch_losses[epoch][3])
                print('Loss Ground Channel: ', epoch_losses[epoch][4])
                print('Loss Objects Channel: ', epoch_losses[epoch][5])
                print('Loss Buildings Channel: ', epoch_losses[epoch][6])
                print('Loss Vegetation Channel: ', epoch_losses[epoch][7])
                print('Loss Sky Channel: ', epoch_losses[epoch][8])

            if self.saving == True:
                saver = tf.train.Saver()
                saver.save(sess,self.FLAGS.train_dir+'/pretrained1.ckpt')
                print('SAVED MODEL')

    def pretrain_red_channel(self):

        red_pred = self.AE_red(self.input_red)
        cost_red = tf.nn.l2_loss(red_pred-self.label_red)
        loss = tf.nn.l2_loss(red_pred-self.label_red)

        regularizer = tf.contrib.layers.l2_regularizer(scale=0.0005)
        reg_red = tf.contrib.layers.apply_regularization(regularizer,weights_list=[self.red_ec_layer['weights'],
                                                                                   self.red_dc_layer['weights'],
                                                                                   self.red_ec_layer['bias'],
                                                                                   self.red_dc_layer['bias']])
        cost_red += reg_red

        epoch_loss = tf.Variable(0.0,name='epoch_loss',trainable=False)
        val_loss = tf.Variable(0.0,name='val_loss',trainable=False)

        sum_epoch_loss = tf.summary.scalar('Epoch Loss Red Channel',epoch_loss)
        sum_val_loss = tf.summary.scalar('Validation Loss Red Channel',val_loss)

        if self.decay == 'constant':
            learning_rate = 0.0001

        if self.decay == 'piecewise':
            global_step = tf.Variable(0,trainable=False)
            boundaries = [10000,100000,1000000]
            rates = [0.001,0.0001,0.00001,0.000001]
            learning_rate = tf.train.piecewise_constant(global_step,boundaries,rates)

        if self.decay == 'exponential':
            global_step = tf.Variable(0,trainable=False)
            base_lr = 0.01
            learning_rate = tf.train.exponential_decay(base_lr,global_step,1000,0.9)

        opt_red = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_red)

        # validation objects
        validations = np.arange(0,self.n_validation_data)
        set_val = np.random.choice(validations,self.n_training_validations,replace=False)

        epoch_loss_reset = epoch_loss.assign(0)
        epoch_loss_update = epoch_loss.assign_add(cost_red)

        loss_val_reset = val_loss.assign(0)
        loss_val_update = val_loss.assign_add(loss/1080.)




        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.per_process_gpu_memory_fraction = 0.5



        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()

            train_writer1 = tf.summary.FileWriter(self.FLAGS.logs_dir,sess.graph)
            sess.run(tf.global_variables_initializer())

            n_batches = int(len(self.imr_train)/self.batch_size)

            tf.get_default_graph().finalize()

            for epoch in range(0,self.hm_epochs):
                sess.run(epoch_loss_reset)
                time1 = datetime.datetime.now()

                for _ in range(0,n_batches):

                    imr_batch = self.imr_train[_*self.batch_size:(_+1)*self.batch_size,:]

                    imr_in = pretraining_input_distortion(copy(imr_batch))

                    feed_dict = {self.input_red:imr_in,
                                 self.label_red:imr_batch}

                    _, l = sess.run([opt_red, epoch_loss_update], feed_dict=feed_dict)


                sum_train = sess.run(sum_epoch_loss)
                train_writer1.add_summary(sum_train,epoch)

                print('----------------------------------------------------------------')
                print('Epoch', epoch, 'completed out of', self.hm_epochs)
                print('Training Loss (per epoch): ', sess.run(epoch_loss.value()))

                sess.run(loss_val_reset)

                for i in set_val:
                    imr_label = self.imr_val[i]
                    imr_in = pretraining_input_distortion(copy(imr_label),singleframe=True)

                    feed_dict_val = {self.input_red:imr_in,
                                     self.label_red:[imr_label]}

                    im_pred,c_val = sess.run([red_pred,loss_val_update],feed_dict=feed_dict_val)

                sum_val = sess.run(sum_val_loss)
                train_writer1.add_summary(sum_val,epoch)
                print('Validation Loss (per pixel): ', sess.run(val_loss.value())/set_val.shape[0])
                time2 = datetime.datetime.now()
                delta = time2-time1
                print('Epoch Time [seconds]:', delta.seconds)
                print('-----------------------------------------------------------------')


            if self.saving == True:
                saver.save(sess,self.FLAGS.train_dir+'/pretrained_red.ckpt')
                print('SAVED MODEL')

    def pretrain_green_channel(self):

        pred = self.AE_green(self.input_green)
        cost = tf.nn.l2_loss(pred-self.label_green)
        loss = tf.nn.l2_loss(pred-self.label_green)

        regularizer = tf.contrib.layers.l2_regularizer(scale=1e-05)
        reg = tf.contrib.layers.apply_regularization(regularizer,weights_list=[self.green_ec_layer['weights'],
                                                                                    self.green_dc_layer['weights'],
                                                                                   self.green_ec_layer['bias'],
                                                                                   self.green_dc_layer['bias']])
        cost += reg

        epoch_loss = tf.Variable(0.0,name='epoch_loss',trainable=False)
        val_loss = tf.Variable(0.0,name='val_loss',trainable=False)

        sum_epoch_loss = tf.summary.scalar('Epoch Loss Green Channel',epoch_loss)
        sum_val_loss = tf.summary.scalar('Validation Loss Green Channel',val_loss)

        if self.decay == 'constant':
            learning_rate = 0.0001

        if self.decay == 'piecewise':
            global_step = tf.Variable(0,trainable=False)
            boundaries = [10000,100000,1000000]
            rates = [0.001,0.0001,0.00001,0.000001]
            learning_rate = tf.train.piecewise_constant(global_step,boundaries,rates)

        if self.decay == 'exponential':
            global_step = tf.Variable(0,trainable=False)
            base_lr = 0.01
            learning_rate = tf.train.exponential_decay(base_lr,global_step,1000,0.9)

        opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        # validation objects
        validations = np.arange(0,self.n_validation_data)
        set_val = np.random.choice(validations,self.n_training_validations,replace=False)

        epoch_loss_reset = epoch_loss.assign(0)
        epoch_loss_update = epoch_loss.assign_add(cost)

        loss_val_reset = val_loss.assign(0)
        loss_val_update = val_loss.assign_add(loss/1080.)


        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.per_process_gpu_memory_fraction = 0.5


        with tf.Session(config=config) as sess:

            saver = tf.train.Saver()

            train_writer1 = tf.summary.FileWriter(self.FLAGS.logs_dir,sess.graph)
            sess.run(tf.global_variables_initializer())

            n_batches = int(len(self.img_train)/self.batch_size)

            tf.get_default_graph().finalize()

            for epoch in range(0,self.hm_epochs):
                sess.run(epoch_loss_reset)
                time1 = datetime.datetime.now()

                for _ in range(0,n_batches):

                    img_batch = self.img_train[_*self.batch_size:(_+1)*self.batch_size,:]

                    img_in = pretraining_input_distortion(copy(img_batch))

                    feed_dict = {self.input_green:img_in,
                                 self.label_green:img_batch}

                    _, l = sess.run([opt, epoch_loss_update], feed_dict=feed_dict)


                sum_train = sess.run(sum_epoch_loss)
                train_writer1.add_summary(sum_train,epoch)

                print('----------------------------------------------------------------')
                print('Epoch', epoch, 'completed out of', self.hm_epochs)
                print('Training Loss (per epoch): ', sess.run(epoch_loss.value()))

                sess.run(loss_val_reset)

                for i in set_val:
                    img_label = self.img_val[i]
                    img_in = pretraining_input_distortion(copy(img_label),singleframe=True)

                    feed_dict_val = {self.input_green:img_in,
                                     self.label_green:[img_label]}

                    im_pred,c_val = sess.run([pred,loss_val_update],feed_dict=feed_dict_val)

                sum_val = sess.run(sum_val_loss)
                train_writer1.add_summary(sum_val,epoch)
                print('Validation Loss (per pixel): ', sess.run(val_loss.value())/set_val.shape[0])
                time2 = datetime.datetime.now()
                delta = time2-time1
                print('Epoch Time [seconds]:', delta.seconds)
                print('-----------------------------------------------------------------')


            if self.saving == True:
                saver.save(sess,self.FLAGS.train_dir+'/pretrained_green.ckpt')
                print('SAVED MODEL')

    def pretrain_blue_channel(self):

        pred = self.AE_blue(self.input_blue)
        cost = tf.nn.l2_loss(pred-self.label_blue)
        loss = tf.nn.l2_loss(pred-self.label_blue)

        regularizer = tf.contrib.layers.l2_regularizer(scale=1e-05)
        reg = tf.contrib.layers.apply_regularization(regularizer,weights_list=[self.blue_ec_layer['weights'],
                                                                               self.blue_dc_layer['weights'],
                                                                               self.blue_ec_layer['bias'],
                                                                               self.blue_dc_layer['bias']])
        cost += reg

        epoch_loss = tf.Variable(0.0,name='epoch_loss',trainable=False)
        val_loss = tf.Variable(0.0,name='val_loss',trainable=False)

        sum_epoch_loss = tf.summary.scalar('Epoch Loss Blue Channel',epoch_loss)
        sum_val_loss = tf.summary.scalar('Validation Loss Blue Channel',val_loss)

        if self.decay == 'constant':
            learning_rate = 0.0001

        if self.decay == 'piecewise':
            global_step = tf.Variable(0,trainable=False)
            boundaries = [10000,100000,1000000]
            rates = [0.001,0.0001,0.00001,0.000001]
            learning_rate = tf.train.piecewise_constant(global_step,boundaries,rates)

        if self.decay == 'exponential':
            global_step = tf.Variable(0,trainable=False)
            base_lr = 0.01
            learning_rate = tf.train.exponential_decay(base_lr,global_step,1000,0.9)

        opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        # validation objects
        validations = np.arange(0,self.n_validation_data)
        set_val = np.random.choice(validations,self.n_training_validations,replace=False)

        epoch_loss_reset = epoch_loss.assign(0)
        epoch_loss_update = epoch_loss.assign_add(cost)

        loss_val_reset = val_loss.assign(0)
        loss_val_update = val_loss.assign_add(loss/1080.)


        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.per_process_gpu_memory_fraction = 0.5

        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()

            train_writer1 = tf.summary.FileWriter(self.FLAGS.logs_dir,sess.graph)
            sess.run(tf.global_variables_initializer())

            n_batches = int(len(self.imb_train)/self.batch_size)

            tf.get_default_graph().finalize()

            for epoch in range(0,self.hm_epochs):
                sess.run(epoch_loss_reset)
                time1 = datetime.datetime.now()

                for _ in range(0,n_batches):

                    imb_batch = self.imb_train[_*self.batch_size:(_+1)*self.batch_size,:]

                    imb_in = pretraining_input_distortion(copy(imb_batch))

                    feed_dict = {self.input_blue:imb_in,
                                 self.label_blue:imb_batch}

                    _, l = sess.run([opt, epoch_loss_update], feed_dict=feed_dict)


                sum_train = sess.run(sum_epoch_loss)
                train_writer1.add_summary(sum_train,epoch)

                print('----------------------------------------------------------------')
                print('Epoch', epoch, 'completed out of', self.hm_epochs)
                print('Training Loss (per epoch): ', sess.run(epoch_loss.value()))

                sess.run(loss_val_reset)

                for i in set_val:
                    imb_label = self.imb_val[i]
                    imb_in = pretraining_input_distortion(copy(imb_label),singleframe=True)

                    feed_dict_val = {self.input_blue:imb_in,
                                     self.label_blue:[imb_label]}

                    im_pred,c_val = sess.run([pred,loss_val_update],feed_dict=feed_dict_val)

                sum_val = sess.run(sum_val_loss)
                train_writer1.add_summary(sum_val,epoch)
                print('Validation Loss (per pixel): ', sess.run(val_loss.value())/set_val.shape[0])
                time2 = datetime.datetime.now()
                delta = time2-time1
                print('Epoch Time [seconds]:', delta.seconds)
                print('-----------------------------------------------------------------')


            if self.saving == True:
                saver.save(sess,self.FLAGS.train_dir+'/pretrained_blue.ckpt')
                print('SAVED MODEL')

    def pretrain_depth_channel(self):

        print('Depth Pretraining')

        pred = self.AE_depth(self.input_depth)
        cost = tf.nn.l2_loss(tf.multiply(self.depth_mask,pred)-tf.multiply(self.depth_mask,self.label_depth)) + \
               1000*tf.nn.l2_loss(tf.multiply(self.depth_loss_mask,pred)-tf.multiply(self.depth_loss_mask,self.label_depth))
        loss = tf.nn.l2_loss(tf.multiply(self.depth_mask,pred)-tf.multiply(self.depth_mask,self.label_depth)) + \
               1000*tf.nn.l2_loss(tf.multiply(self.depth_loss_mask,pred)-tf.multiply(self.depth_loss_mask,self.label_depth))

        regularizer = tf.contrib.layers.l2_regularizer(scale=1e-05)
        reg = tf.contrib.layers.apply_regularization(regularizer,weights_list=[self.depth_ec_layer['weights'],
                                                                               self.depth_dc_layer['weights'],
                                                                               self.depth_ec_layer['bias'],
                                                                               self.depth_dc_layer['bias']])
        cost += reg

        epoch_loss = tf.Variable(0.0,name='epoch_loss',trainable=False)
        val_loss = tf.Variable(0.0,name='val_loss',trainable=False)

        sum_epoch_loss = tf.summary.scalar('Epoch Loss Depth Channel',epoch_loss)
        sum_val_loss = tf.summary.scalar('Validation Loss Depth Channel',val_loss)

        if self.decay == 'constant':
            learning_rate = 0.0001

        if self.decay == 'piecewise':
            global_step = tf.Variable(0,trainable=False)
            boundaries = [10000,100000,1000000]
            rates = [0.001,0.0001,0.00001,0.000001]
            learning_rate = tf.train.piecewise_constant(global_step,boundaries,rates)

        if self.decay == 'exponential':
            global_step = tf.Variable(0,trainable=False)
            base_lr = 0.01
            learning_rate = tf.train.exponential_decay(base_lr,global_step,1000,0.9)

        opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        # validation objects
        validations = np.arange(0,self.n_validation_data)
        set_val = np.random.choice(validations,self.n_training_validations,replace=False)

        epoch_loss_reset = epoch_loss.assign(0)
        epoch_loss_update = epoch_loss.assign_add(cost)

        loss_val_reset = val_loss.assign(0)
        loss_val_update = val_loss.assign_add(loss/1080.)


        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.per_process_gpu_memory_fraction = 0.5

        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()

            train_writer1 = tf.summary.FileWriter(self.FLAGS.logs_dir,sess.graph)
            sess.run(tf.global_variables_initializer())

            n_batches = int(len(self.imb_train)/self.batch_size)

            tf.get_default_graph().finalize()

            for epoch in range(0,self.hm_epochs):
                sess.run(epoch_loss_reset)
                time1 = datetime.datetime.now()

                for _ in range(0,n_batches):

                    batch = self.depth_train[_*self.batch_size:(_+1)*self.batch_size,:]
                    batch_mask = self.depth_mask_train[_*self.batch_size:(_+1)*self.batch_size,:]
                    batch_loss_mask = self.depth_loss_mask_train[_*self.batch_size:(_+1)*self.batch_size,:]

                    depth_in = pretraining_input_distortion(copy(batch))

                    feed_dict = {self.input_depth:depth_in,
                                 self.label_depth:batch,
                                 self.depth_mask:batch_mask,
                                 self.depth_loss_mask:batch_loss_mask}

                    _, l = sess.run([opt, epoch_loss_update], feed_dict=feed_dict)


                sum_train = sess.run(sum_epoch_loss)
                train_writer1.add_summary(sum_train,epoch)

                print('----------------------------------------------------------------')
                print('Epoch', epoch, 'completed out of', self.hm_epochs)
                print('Training Loss (per epoch): ', sess.run(epoch_loss.value()))

                sess.run(loss_val_reset)

                for i in set_val:
                    depth_label = self.depth_val[i]
                    depth_mask = self.depth_mask_val[i]
                    depth_loss_mask = self.depth_loss_mask_val[i]
                    depth_in = pretraining_input_distortion(copy(depth_label),singleframe=True)

                    feed_dict_val = {self.input_depth:depth_in,
                                     self.label_depth:[depth_label],
                                     self.depth_mask:[depth_mask],
                                     self.depth_loss_mask:[depth_loss_mask]}

                    im_pred,c_val = sess.run([pred,loss_val_update],feed_dict=feed_dict_val)

                # uncomment when not running on gpu
                #if epoch%10==0:
                     #print_validation_frames(depth_in,im_pred,depth_label,channel='depth',shape=(60,18))

                sum_val = sess.run(sum_val_loss)
                train_writer1.add_summary(sum_val,epoch)
                print('Validation Loss (per pixel): ', sess.run(val_loss.value())/set_val.shape[0])
                time2 = datetime.datetime.now()
                delta = time2-time1
                print('Epoch Time [seconds]:', delta.seconds)
                print('-----------------------------------------------------------------')


            if self.saving == True:
                saver.save(sess,self.FLAGS.train_dir+'/pretrained_depth.ckpt')
                print('SAVED MODEL')

    def pretrain_gnd_channel(self):

        pred = self.AE_gnd(self.input_gnd)
        cost = tf.nn.l2_loss(pred-self.label_gnd)
        loss = tf.nn.l2_loss(pred-self.label_gnd)

        regularizer = tf.contrib.layers.l2_regularizer(scale=1e-05)
        reg = tf.contrib.layers.apply_regularization(regularizer,weights_list=[self.gnd_ec_layer['weights'],
                                                                                self.gnd_dc_layer['weights'],
                                                                                self.gnd_ec_layer['bias'],
                                                                                self.gnd_dc_layer['bias']])
        cost += reg

        epoch_loss = tf.Variable(0.0,name='epoch_loss',trainable=False)
        val_loss = tf.Variable(0.0,name='val_loss',trainable=False)

        sum_epoch_loss = tf.summary.scalar('Epoch Loss Ground Channel',epoch_loss)
        sum_val_loss = tf.summary.scalar('Validation Loss Ground Channel',val_loss)

        if self.decay == 'constant':
            learning_rate = 0.0001

        if self.decay == 'piecewise':
            global_step = tf.Variable(0,trainable=False)
            boundaries = [10000,100000,1000000]
            rates = [0.001,0.0001,0.00001,0.000001]
            learning_rate = tf.train.piecewise_constant(global_step,boundaries,rates)

        if self.decay == 'exponential':
            global_step = tf.Variable(0,trainable=False)
            base_lr = 0.01
            learning_rate = tf.train.exponential_decay(base_lr,global_step,1000,0.9)

        opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        # validation objects
        validations = np.arange(0,self.n_validation_data)
        set_val = np.random.choice(validations,self.n_training_validations,replace=False)

        epoch_loss_reset = epoch_loss.assign(0)
        epoch_loss_update = epoch_loss.assign_add(cost)

        loss_val_reset = val_loss.assign(0)
        loss_val_update = val_loss.assign_add(loss/1080.)


        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.per_process_gpu_memory_fraction = 0.5


        with tf.Session(config=config) as sess:

            saver = tf.train.Saver()

            train_writer1 = tf.summary.FileWriter(self.FLAGS.logs_dir,sess.graph)
            sess.run(tf.global_variables_initializer())

            n_batches = int(len(self.gnd_train)/self.batch_size)

            #tf.get_default_graph().finalize()

            for epoch in range(0,self.hm_epochs):
                sess.run(epoch_loss_reset)
                time1 = datetime.datetime.now()

                for _ in range(0,n_batches):

                    batch = self.gnd_train[_*self.batch_size:(_+1)*self.batch_size,:]

                    inp = pretraining_input_distortion(copy(batch))

                    feed_dict = {self.input_gnd:inp,
                                 self.label_gnd:batch}

                    _, l = sess.run([opt, epoch_loss_update], feed_dict=feed_dict)


                sum_train = sess.run(sum_epoch_loss)
                train_writer1.add_summary(sum_train,epoch)

                print('----------------------------------------------------------------')
                print('Epoch', epoch, 'completed out of', self.hm_epochs)
                print('Training Loss (per epoch): ', sess.run(epoch_loss.value()))

                sess.run(loss_val_reset)

                for i in set_val:
                    label = self.gnd_val[i]
                    inp = pretraining_input_distortion(copy(label),singleframe=True)

                    feed_dict_val = {self.input_gnd:inp,
                                     self.label_gnd:[label]}

                    im_pred,c_val = sess.run([pred,loss_val_update],feed_dict=feed_dict_val)

                sum_val = sess.run(sum_val_loss)
                train_writer1.add_summary(sum_val,epoch)
                print('Validation Loss (per pixel): ', sess.run(val_loss.value())/set_val.shape[0])
                time2 = datetime.datetime.now()
                delta = time2-time1
                print('Epoch Time [seconds]:', delta.seconds)
                print('-----------------------------------------------------------------')


            if self.saving == True:
                saver.save(sess,self.FLAGS.train_dir+'/pretrained_gnd.ckpt')
                print('SAVED MODEL')

    def pretrain_obj_channel(self):

        pred = self.AE_obj(self.input_obj)
        cost = tf.nn.l2_loss(pred-self.label_obj)
        loss = tf.nn.l2_loss(pred-self.label_obj)

        regularizer = tf.contrib.layers.l2_regularizer(scale=1e-05)
        reg = tf.contrib.layers.apply_regularization(regularizer,weights_list=[self.obj_ec_layer['weights'],
                                                                                self.obj_dc_layer['weights'],
                                                                                self.obj_ec_layer['bias'],
                                                                                self.obj_dc_layer['bias']])
        cost += reg

        epoch_loss = tf.Variable(0.0,name='epoch_loss',trainable=False)
        val_loss = tf.Variable(0.0,name='val_loss',trainable=False)

        sum_epoch_loss = tf.summary.scalar('Epoch Loss Object Channel',epoch_loss)
        sum_val_loss = tf.summary.scalar('Validation Loss Object Channel',val_loss)

        if self.decay == 'constant':
            learning_rate = 0.0001

        if self.decay == 'piecewise':
            global_step = tf.Variable(0,trainable=False)
            boundaries = [10000,100000,1000000]
            rates = [0.001,0.0001,0.00001,0.000001]
            learning_rate = tf.train.piecewise_constant(global_step,boundaries,rates)

        if self.decay == 'exponential':
            global_step = tf.Variable(0,trainable=False)
            base_lr = 0.01
            learning_rate = tf.train.exponential_decay(base_lr,global_step,1000,0.9)

        opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        # validation objects
        validations = np.arange(0,self.n_validation_data)
        set_val = np.random.choice(validations,self.n_training_validations,replace=False)

        epoch_loss_reset = epoch_loss.assign(0)
        epoch_loss_update = epoch_loss.assign_add(cost)

        loss_val_reset = val_loss.assign(0)
        loss_val_update = val_loss.assign_add(loss/1080.)


        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.per_process_gpu_memory_fraction = 0.5


        with tf.Session(config=config) as sess:

            saver = tf.train.Saver()

            train_writer1 = tf.summary.FileWriter(self.FLAGS.logs_dir,sess.graph)
            sess.run(tf.global_variables_initializer())

            n_batches = int(len(self.obj_train)/self.batch_size)

            #tf.get_default_graph().finalize()

            for epoch in range(0,self.hm_epochs):
                sess.run(epoch_loss_reset)
                time1 = datetime.datetime.now()

                for _ in range(0,n_batches):

                    batch = self.obj_train[_*self.batch_size:(_+1)*self.batch_size,:]

                    inp = pretraining_input_distortion(copy(batch))

                    feed_dict = {self.input_obj:inp,
                                 self.label_obj:batch}

                    _, l = sess.run([opt, epoch_loss_update], feed_dict=feed_dict)


                sum_train = sess.run(sum_epoch_loss)
                train_writer1.add_summary(sum_train,epoch)

                print('----------------------------------------------------------------')
                print('Epoch', epoch, 'completed out of', self.hm_epochs)
                print('Training Loss (per epoch): ', sess.run(epoch_loss.value()))

                sess.run(loss_val_reset)

                for i in set_val:
                    label = self.obj_val[i]
                    inp = pretraining_input_distortion(copy(label),singleframe=True)

                    feed_dict_val = {self.input_obj:inp,
                                     self.label_obj:[label]}

                    im_pred,c_val = sess.run([pred,loss_val_update],feed_dict=feed_dict_val)

                sum_val = sess.run(sum_val_loss)
                train_writer1.add_summary(sum_val,epoch)
                print('Validation Loss (per pixel): ', sess.run(val_loss.value())/set_val.shape[0])
                time2 = datetime.datetime.now()
                delta = time2-time1
                print('Epoch Time [seconds]:', delta.seconds)
                print('-----------------------------------------------------------------')


            if self.saving == True:
                saver.save(sess,self.FLAGS.train_dir+'/pretrained_obj.ckpt')
                print('SAVED MODEL')

    def pretrain_bld_channel(self):

        pred = self.AE_bld(self.input_bld)
        cost = tf.nn.l2_loss(pred-self.label_bld)
        loss = tf.nn.l2_loss(pred-self.label_bld)

        regularizer = tf.contrib.layers.l2_regularizer(scale=1e-05)
        reg = tf.contrib.layers.apply_regularization(regularizer,weights_list=[self.bld_ec_layer['weights'],
                                                                                self.bld_dc_layer['weights'],
                                                                                self.bld_ec_layer['bias'],
                                                                                self.bld_dc_layer['bias']])
        cost += reg

        epoch_loss = tf.Variable(0.0,name='epoch_loss',trainable=False)
        val_loss = tf.Variable(0.0,name='val_loss',trainable=False)

        sum_epoch_loss = tf.summary.scalar('Epoch Loss Building Channel',epoch_loss)
        sum_val_loss = tf.summary.scalar('Validation Loss Building Channel',val_loss)

        if self.decay == 'constant':
            learning_rate = 0.0001

        if self.decay == 'piecewise':
            global_step = tf.Variable(0,trainable=False)
            boundaries = [10000,100000,1000000]
            rates = [0.001,0.0001,0.00001,0.000001]
            learning_rate = tf.train.piecewise_constant(global_step,boundaries,rates)

        if self.decay == 'exponential':
            global_step = tf.Variable(0,trainable=False)
            base_lr = 0.01
            learning_rate = tf.train.exponential_decay(base_lr,global_step,1000,0.9)

        opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        # validation objects
        validations = np.arange(0,self.n_validation_data)
        set_val = np.random.choice(validations,self.n_training_validations,replace=False)

        epoch_loss_reset = epoch_loss.assign(0)
        epoch_loss_update = epoch_loss.assign_add(cost)

        loss_val_reset = val_loss.assign(0)
        loss_val_update = val_loss.assign_add(loss/1080.)


        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.per_process_gpu_memory_fraction = 0.5


        with tf.Session(config=config) as sess:

            saver = tf.train.Saver()

            train_writer1 = tf.summary.FileWriter(self.FLAGS.logs_dir,sess.graph)
            sess.run(tf.global_variables_initializer())

            n_batches = int(len(self.bld_train)/self.batch_size)

            #tf.get_default_graph().finalize()

            for epoch in range(0,self.hm_epochs):
                sess.run(epoch_loss_reset)
                time1 = datetime.datetime.now()

                for _ in range(0,n_batches):

                    batch = self.bld_train[_*self.batch_size:(_+1)*self.batch_size,:]

                    inp = pretraining_input_distortion(copy(batch))

                    feed_dict = {self.input_bld:inp,
                                 self.label_bld:batch}

                    _, l = sess.run([opt, epoch_loss_update], feed_dict=feed_dict)


                sum_train = sess.run(sum_epoch_loss)
                train_writer1.add_summary(sum_train,epoch)

                print('----------------------------------------------------------------')
                print('Epoch', epoch, 'completed out of', self.hm_epochs)
                print('Training Loss (per epoch): ', sess.run(epoch_loss.value()))

                sess.run(loss_val_reset)

                for i in set_val:
                    label = self.bld_val[i]
                    inp = pretraining_input_distortion(copy(label),singleframe=True)

                    feed_dict_val = {self.input_bld:inp,
                                     self.label_bld:[label]}

                    im_pred,c_val = sess.run([pred,loss_val_update],feed_dict=feed_dict_val)

                sum_val = sess.run(sum_val_loss)
                train_writer1.add_summary(sum_val,epoch)
                print('Validation Loss (per pixel): ', sess.run(val_loss.value())/set_val.shape[0])
                time2 = datetime.datetime.now()
                delta = time2-time1
                print('Epoch Time [seconds]:', delta.seconds)
                print('-----------------------------------------------------------------')


            if self.saving == True:
                saver.save(sess,self.FLAGS.train_dir+'/pretrained_bld.ckpt')
                print('SAVED MODEL')

    def pretrain_veg_channel(self):

        pred = self.AE_veg(self.input_veg)
        cost = tf.nn.l2_loss(pred-self.label_veg)
        loss = tf.nn.l2_loss(pred-self.label_veg)

        regularizer = tf.contrib.layers.l2_regularizer(scale=1e-05)
        reg = tf.contrib.layers.apply_regularization(regularizer,weights_list=[self.veg_ec_layer['weights'],
                                                                                self.veg_dc_layer['weights'],
                                                                                self.veg_ec_layer['bias'],
                                                                                self.veg_dc_layer['bias']])
        cost += reg

        epoch_loss = tf.Variable(0.0,name='epoch_loss',trainable=False)
        val_loss = tf.Variable(0.0,name='val_loss',trainable=False)

        sum_epoch_loss = tf.summary.scalar('Epoch Loss Vegetation Channel',epoch_loss)
        sum_val_loss = tf.summary.scalar('Validation Loss Vegetation Channel',val_loss)

        if self.decay == 'constant':
            learning_rate = 0.0001

        if self.decay == 'piecewise':
            global_step = tf.Variable(0,trainable=False)
            boundaries = [10000,100000,1000000]
            rates = [0.001,0.0001,0.00001,0.000001]
            learning_rate = tf.train.piecewise_constant(global_step,boundaries,rates)

        if self.decay == 'exponential':
            global_step = tf.Variable(0,trainable=False)
            base_lr = 0.01
            learning_rate = tf.train.exponential_decay(base_lr,global_step,1000,0.9)

        opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        # validation objects
        validations = np.arange(0,self.n_validation_data)
        set_val = np.random.choice(validations,self.n_training_validations,replace=False)

        epoch_loss_reset = epoch_loss.assign(0)
        epoch_loss_update = epoch_loss.assign_add(cost)

        loss_val_reset = val_loss.assign(0)
        loss_val_update = val_loss.assign_add(loss/1080.)


        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.per_process_gpu_memory_fraction = 0.5


        with tf.Session(config=config) as sess:

            saver = tf.train.Saver()

            train_writer1 = tf.summary.FileWriter(self.FLAGS.logs_dir,sess.graph)
            sess.run(tf.global_variables_initializer())

            n_batches = int(len(self.veg_train)/self.batch_size)

            #tf.get_default_graph().finalize()

            for epoch in range(0,self.hm_epochs):
                sess.run(epoch_loss_reset)
                time1 = datetime.datetime.now()

                for _ in range(0,n_batches):

                    batch = self.veg_train[_*self.batch_size:(_+1)*self.batch_size,:]

                    inp = pretraining_input_distortion(copy(batch))

                    feed_dict = {self.input_veg:inp,
                                 self.label_veg:batch}

                    _, l = sess.run([opt, epoch_loss_update], feed_dict=feed_dict)


                sum_train = sess.run(sum_epoch_loss)
                train_writer1.add_summary(sum_train,epoch)

                print('----------------------------------------------------------------')
                print('Epoch', epoch, 'completed out of', self.hm_epochs)
                print('Training Loss (per epoch): ', sess.run(epoch_loss.value()))

                sess.run(loss_val_reset)

                for i in set_val:
                    label = self.veg_val[i]
                    inp = pretraining_input_distortion(copy(label),singleframe=True)

                    feed_dict_val = {self.input_veg:inp,
                                     self.label_veg:[label]}

                    im_pred,c_val = sess.run([pred,loss_val_update],feed_dict=feed_dict_val)

                sum_val = sess.run(sum_val_loss)
                train_writer1.add_summary(sum_val,epoch)
                print('Validation Loss (per pixel): ', sess.run(val_loss.value())/set_val.shape[0])
                time2 = datetime.datetime.now()
                delta = time2-time1
                print('Epoch Time [seconds]:', delta.seconds)
                print('-----------------------------------------------------------------')


            if self.saving == True:
                saver.save(sess,self.FLAGS.train_dir+'/pretrained_veg.ckpt')
                print('SAVED MODEL')

    def pretrain_sky_channel(self):

        pred = self.AE_sky(self.input_sky)
        cost = tf.nn.l2_loss(pred-self.label_sky)
        loss = tf.nn.l2_loss(pred-self.label_sky)

        regularizer = tf.contrib.layers.l2_regularizer(scale=1e-05)
        reg = tf.contrib.layers.apply_regularization(regularizer,weights_list=[self.sky_ec_layer['weights'],
                                                                                self.sky_dc_layer['weights'],
                                                                                self.sky_ec_layer['bias'],
                                                                                self.sky_dc_layer['bias']])
        cost += reg

        epoch_loss = tf.Variable(0.0,name='epoch_loss',trainable=False)
        val_loss = tf.Variable(0.0,name='val_loss',trainable=False)

        sum_epoch_loss = tf.summary.scalar('Epoch Loss Sky Channel',epoch_loss)
        sum_val_loss = tf.summary.scalar('Validation Loss Sky Channel',val_loss)

        if self.decay == 'constant':
            learning_rate = 0.0001

        if self.decay == 'piecewise':
            global_step = tf.Variable(0,trainable=False)
            boundaries = [10000,100000,1000000]
            rates = [0.001,0.0001,0.00001,0.000001]
            learning_rate = tf.train.piecewise_constant(global_step,boundaries,rates)

        if self.decay == 'exponential':
            global_step = tf.Variable(0,trainable=False)
            base_lr = 0.01
            learning_rate = tf.train.exponential_decay(base_lr,global_step,1000,0.9)

        opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        # validation objects
        validations = np.arange(0,self.n_validation_data)
        set_val = np.random.choice(validations,self.n_training_validations,replace=False)

        epoch_loss_reset = epoch_loss.assign(0)
        epoch_loss_update = epoch_loss.assign_add(cost)

        loss_val_reset = val_loss.assign(0)
        loss_val_update = val_loss.assign_add(loss/1080.)


        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.per_process_gpu_memory_fraction = 0.5


        with tf.Session(config=config) as sess:

            saver = tf.train.Saver()

            train_writer1 = tf.summary.FileWriter(self.FLAGS.logs_dir,sess.graph)
            sess.run(tf.global_variables_initializer())

            n_batches = int(len(self.sky_train)/self.batch_size)

            #tf.get_default_graph().finalize()

            for epoch in range(0,self.hm_epochs):
                sess.run(epoch_loss_reset)
                time1 = datetime.datetime.now()

                for _ in range(0,n_batches):

                    batch = self.sky_train[_*self.batch_size:(_+1)*self.batch_size,:]

                    inp = pretraining_input_distortion(copy(batch))

                    feed_dict = {self.input_sky:inp,
                                 self.label_sky:batch}

                    _, l = sess.run([opt, epoch_loss_update], feed_dict=feed_dict)


                sum_train = sess.run(sum_epoch_loss)
                train_writer1.add_summary(sum_train,epoch)

                print('----------------------------------------------------------------')
                print('Epoch', epoch, 'completed out of', self.hm_epochs)
                print('Training Loss (per epoch): ', sess.run(epoch_loss.value()))

                sess.run(loss_val_reset)

                for i in set_val:
                    label = self.sky_val[i]
                    inp = pretraining_input_distortion(copy(label),singleframe=True)

                    feed_dict_val = {self.input_sky:inp,
                                     self.label_sky:[label]}

                    im_pred,c_val = sess.run([pred,loss_val_update],feed_dict=feed_dict_val)

                sum_val = sess.run(sum_val_loss)
                train_writer1.add_summary(sum_val,epoch)
                print('Validation Loss (per pixel): ', sess.run(val_loss.value())/set_val.shape[0])
                time2 = datetime.datetime.now()
                delta = time2-time1
                print('Epoch Time [seconds]:', delta.seconds)
                print('-----------------------------------------------------------------')


            if self.saving == True:
                saver.save(sess,self.FLAGS.train_dir+'/pretrained_sky.ckpt')
                print('SAVED MODEL')

    def pretrain_shared_semantics(self):

        gnd_pred = self.AE_gnd(self.input_gnd)
        obj_pred = self.AE_obj(self.input_obj)
        bld_pred = self.AE_bld(self.input_bld)
        veg_pred = self.AE_veg(self.input_veg)
        sky_pred = self.AE_sky(self.input_sky)

        gnd_pred,obj_pred,bld_pred,veg_pred,sky_pred = self.AE_sem(self.input_gnd,
                                                                   self.input_obj,
                                                                   self.input_bld,
                                                                   self.input_veg,
                                                                   self.input_sky)

        cost = tf.nn.l2_loss(gnd_pred-self.label_gnd) + \
               tf.nn.l2_loss(obj_pred-self.label_obj) + \
               tf.nn.l2_loss(bld_pred-self.label_bld) + \
               tf.nn.l2_loss(veg_pred-self.label_veg) + \
               tf.nn.l2_loss(sky_pred-self.label_sky)


        loss = tf.nn.l2_loss(gnd_pred-self.label_gnd) + \
               tf.nn.l2_loss(obj_pred-self.label_obj) + \
               tf.nn.l2_loss(bld_pred-self.label_bld) + \
               tf.nn.l2_loss(veg_pred-self.label_veg) + \
               tf.nn.l2_loss(sky_pred-self.label_sky)

        regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)
        reg_sem = tf.contrib.layers.apply_regularization(regularizer,weights_list=[self.gnd_ec_layer['weights'],
                                                                                    self.gnd_ec_layer['bias'],
                                                                                    self.obj_ec_layer['weights'],
                                                                                    self.obj_ec_layer['bias'],
                                                                                    self.bld_ec_layer['weights'],
                                                                                    self.bld_ec_layer['bias'],
                                                                                    self.veg_ec_layer['weights'],
                                                                                    self.veg_ec_layer['bias'],
                                                                                    self.sky_ec_layer['weights'],
                                                                                    self.sky_ec_layer['bias'],
                                                                                    self.sem_ec_layer['weights'],
                                                                                    self.sem_ec_layer['bias'],
                                                                                    self.sem_dc_layer['weights'],
                                                                                    self.sem_dc_layer['bias'],
                                                                                    self.gnd_dc_layer['weights'],
                                                                                    self.gnd_dc_layer['bias'],
                                                                                    self.obj_dc_layer['weights'],
                                                                                    self.obj_dc_layer['bias'],
                                                                                    self.bld_dc_layer['weights'],
                                                                                    self.bld_dc_layer['bias'],
                                                                                    self.veg_dc_layer['weights'],
                                                                                    self.veg_dc_layer['bias'],
                                                                                    self.sky_dc_layer['weights'],
                                                                                    self.sky_dc_layer['bias'],])
        cost += reg_sem

        epoch_loss = tf.Variable(0.0,name='epoch_loss',trainable=False)
        val_loss = tf.Variable(0.0,name='val_loss',trainable=False)

        sum_epoch_loss = tf.summary.scalar('Epoch Loss Shared Semantics',epoch_loss)
        sum_val_loss = tf.summary.scalar('Validation Loss Shared Semantics',val_loss)

        if self.decay == 'constant':
            learning_rate = 0.0001

        if self.decay == 'piecewise':
            global_step = tf.Variable(0,trainable=False)
            boundaries = [10000,100000,1000000]
            rates = [0.001,0.0001,0.00001,0.000001]
            learning_rate = tf.train.piecewise_constant(global_step,boundaries,rates)

        if self.decay == 'exponential':
            global_step = tf.Variable(0,trainable=False)
            base_lr = 0.01
            learning_rate = tf.train.exponential_decay(base_lr,global_step,1000,0.9)

        opt1 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost,var_list=[self.sem_ec_layer['weights'],
                                                                                           self.sem_ec_layer['bias'],
                                                                                           self.sem_dc_layer['weights'],
                                                                                           self.sem_dc_layer['bias']])
        opt2 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        # validation objects
        validations = np.arange(0,self.n_validation_data)
        set_val = np.random.choice(validations,self.n_training_validations,replace=False)

        epoch_loss_reset = epoch_loss.assign(0)
        epoch_loss_update = epoch_loss.assign_add(cost)

        loss_val_reset = val_loss.assign(0)
        loss_val_update = val_loss.assign_add(loss/1080.)


        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.per_process_gpu_memory_fraction = 0.5



        saver_load_gnd = tf.train.Saver({'gnd_ec_layer_weights':self.gnd_ec_layer['weights'],
                                         'gnd_ec_layer_bias':self.gnd_ec_layer['bias'],
                                         'gnd_dc_layer_weights':self.gnd_dc_layer['weights'],
                                         'gnd_dc_layer_bias':self.gnd_dc_layer['bias']})

        saver_load_obj = tf.train.Saver({'obj_ec_layer_weights':self.obj_ec_layer['weights'],
                                         'obj_ec_layer_bias':self.obj_ec_layer['bias'],
                                         'obj_dc_layer_weights':self.obj_dc_layer['weights'],
                                         'obj_dc_layer_bias':self.obj_dc_layer['bias']})

        saver_load_bld = tf.train.Saver({'bld_ec_layer_weights':self.bld_ec_layer['weights'],
                                         'bld_ec_layer_bias':self.bld_ec_layer['bias'],
                                         'bld_dc_layer_weights':self.bld_dc_layer['weights'],
                                         'bld_dc_layer_bias':self.bld_dc_layer['bias']})

        saver_load_veg = tf.train.Saver({'veg_ec_layer_weights':self.veg_ec_layer['weights'],
                                         'veg_ec_layer_bias':self.veg_ec_layer['bias'],
                                         'veg_dc_layer_weights':self.veg_dc_layer['weights'],
                                         'veg_dc_layer_bias':self.veg_dc_layer['bias']})

        saver_load_sky = tf.train.Saver({'sky_ec_layer_weights':self.sky_ec_layer['weights'],
                                         'sky_ec_layer_bias':self.sky_ec_layer['bias'],
                                         'sky_dc_layer_weights':self.sky_dc_layer['weights'],
                                         'sky_dc_layer_bias':self.sky_dc_layer['bias']})



        saver_save = tf.train.Saver()

        with tf.Session(config=config) as sess:

            sess.run(tf.global_variables_initializer())
            saver_load_gnd.restore(sess,self.FLAGS.train_dir+'/pretrained_gnd.ckpt')
            saver_load_obj.restore(sess,self.FLAGS.train_dir+'/pretrained_obj.ckpt')
            saver_load_bld.restore(sess,self.FLAGS.train_dir+'/pretrained_bld.ckpt')
            saver_load_veg.restore(sess,self.FLAGS.train_dir+'/pretrained_veg.ckpt')
            saver_load_sky.restore(sess,self.FLAGS.train_dir+'/pretrained_sky.ckpt')

            train_writer1 = tf.summary.FileWriter(self.FLAGS.logs_dir,sess.graph)

            n_batches = int(len(self.gnd_train)/self.batch_size)

            tf.get_default_graph().finalize()

            for epoch in range(0,self.hm_epochs):
                sess.run(epoch_loss_reset)
                time1 = datetime.datetime.now()

                for _ in range(0,n_batches):

                    imr_batch = self.imr_train[_*self.batch_size:(_+1)*self.batch_size]
                    img_batch = self.img_train[_*self.batch_size:(_+1)*self.batch_size]
                    imb_batch = self.imb_train[_*self.batch_size:(_+1)*self.batch_size]
                    depth_batch = self.depth_train[_*self.batch_size:(_+1)*self.batch_size]
                    gnd_batch = self.gnd_train[_*self.batch_size:(_+1)*self.batch_size]
                    obj_batch = self.obj_train[_*self.batch_size:(_+1)*self.batch_size]
                    bld_batch = self.bld_train[_*self.batch_size:(_+1)*self.batch_size]
                    veg_batch = self.veg_train[_*self.batch_size:(_+1)*self.batch_size]
                    sky_batch = self.sky_train[_*self.batch_size:(_+1)*self.batch_size]

                    imr_in,img_in,imb_in,depth_in,gnd_in,obj_in,bld_in,veg_in,sky_in = input_distortion(copy(imr_batch),
                                                                                                        copy(img_batch),
                                                                                                        copy(imb_batch),
                                                                                                        copy(depth_batch),
                                                                                                        copy(gnd_batch),
                                                                                                        copy(obj_batch),
                                                                                                        copy(bld_batch),
                                                                                                        copy(veg_batch),
                                                                                                        copy(sky_batch),
                                                                                                        border1=1,
                                                                                                        border2=1,
                                                                                                        resolution=(18,60))

                    feed_dict_sem = {self.input_gnd:gnd_in,
                                     self.label_gnd:gnd_batch,
                                     self.input_obj:obj_in,
                                     self.label_obj:obj_batch,
                                     self.input_bld:bld_in,
                                     self.label_bld:bld_batch,
                                     self.input_veg:veg_in,
                                     self.label_veg:veg_batch,
                                     self.input_sky:sky_in,
                                     self.label_sky:sky_batch}

                    if epoch < 10:
                        _, c = sess.run([opt1,epoch_loss_update],feed_dict=feed_dict_sem)
                    else:
                        _, c = sess.run([opt2,epoch_loss_update],feed_dict=feed_dict_sem)


                sum_train = sess.run(sum_epoch_loss)
                train_writer1.add_summary(sum_train,epoch)
                print('Epoch', epoch, 'completed out of', self.hm_epochs)
                print('Training Loss (per epoch): ', sess.run(epoch_loss.value()))

                sess.run(loss_val_reset)

                for i in set_val:
                    label_imr = self.imr_val[i]
                    label_img = self.img_val[i]
                    label_imb = self.imb_val[i]
                    label_depth = self.depth_val[i]
                    label_gnd = self.gnd_val[i]
                    label_obj = self.obj_val[i]
                    label_bld = self.bld_val[i]
                    label_veg = self.veg_val[i]
                    label_sky = self.sky_val[i]

                    imr_in,img_in,imb_in,depth_in,gnd_in,obj_in,bld_in,veg_in,sky_in = input_distortion(copy(label_imr),
                                                                                                        copy(label_img),
                                                                                                        copy(label_imb),
                                                                                                        copy(label_depth),
                                                                                                        copy(label_gnd),
                                                                                                        copy(label_obj),
                                                                                                        copy(label_bld),
                                                                                                        copy(label_veg),
                                                                                                        copy(label_sky),
                                                                                                        border1=1,
                                                                                                        border2=1,
                                                                                                        resolution=(18,60),
                                                                                                        singleframe=True)


                    feed_dict_val = {self.input_gnd:gnd_in,
                                     self.input_obj:obj_in,
                                     self.input_bld:bld_in,
                                     self.input_veg:veg_in,
                                     self.input_sky:sky_in,
                                     self.label_gnd:[label_gnd],
                                     self.label_obj:[label_obj],
                                     self.label_bld:[label_bld],
                                     self.label_veg:[label_veg],
                                     self.label_sky:[label_sky]}

                    l,l_up = sess.run([loss,loss_val_update],feed_dict=feed_dict_val)

                sum_val = sess.run(sum_val_loss)
                train_writer1.add_summary(sum_val,epoch)
                print('Validation Loss (per pixel): ', sess.run(val_loss.value())/set_val.shape[0])
                time2 = datetime.datetime.now()
                delta = time2-time1
                print('Epoch Time [seconds]:', delta.seconds)
                print('-----------------------------------------------------------------')






            if self.saving==True:
                saver_save.save(sess,self.FLAGS.train_dir+'/pretrained_shared_semantics.ckpt')
                print('SAVED MODEL')



    def validate_depth(self):

        prediction = self.AE_depth(self.input_depth)
        loss = tf.nn.l2_loss(tf.multiply(self.depth_mask,prediction)-tf.multiply(self.depth_mask,self.label_depth))

        load_weights = tf.train.Saver()

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            load_weights.restore(sess,self.FLAGS.train_dir+'/pretrained_depth.ckpt')

            for i in range(0,self.n_validation_data):

                depth_label = self.depth_val[i]
                depth_mask = self.depth_mask_val[i]
                depth_input = pretraining_input_distortion(copy(depth_label),singleframe=True)

                feed_dict = {self.input_depth:depth_input,
                             self.label_depth:[depth_label],
                             self.depth_mask:[depth_mask]}

                depth_pred, l = sess.run([prediction,loss],feed_dict=feed_dict)
                #print_validation_frames(depth_input,depth_pred,depth_label,channel='depth',shape=(60,18))
                print(np.reshape(depth_pred,(18,60)))
                print(np.reshape(depth_label,(18,60)))

                print('Validation Loss:', l/1080.)






pretraining = PretrainingMAE(data_train, data_validate, data_test)

#pretraining.pretrain_red_channel()
#pretraining.pretrain_green_channel()
#pretraining.pretrain_blue_channel()

#pretraining.pretrain_gnd_channel()
#pretraining.pretrain_obj_channel()
#pretraining.pretrain_bld_channel()
#pretraining.pretrain_veg_channel()
#pretraining.pretrain_sky_channel()

#pretraining.pretrain_shared_semantics()

pretraining.pretrain_depth_channel()

#pretraining.validate_depth()
