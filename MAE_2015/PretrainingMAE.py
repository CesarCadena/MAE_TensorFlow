
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os

import sys
sys.path.append('./tools')

from input_distortion import input_distortion,pretraining_input_distortion
from visualization import print_training_frames,print_validation_frames
from basic_routines import horizontal_mirroring
from copy import copy


class PretrainingMAE():

    def __init__(self,data_train,data_validate):

        # storing data
        self.data_train = data_train
        self.data_val = data_validate        

        # training options

        self.batch_size = 100
        self.hm_epochs = 300

        self.input_size = 1080
        self.hidden_size = 1024

        self.saving = True        

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

        now = datetime.datetime.now()

        self.mode = 'pretraining/'

        self.model_folder = 'models/'
        self.logs_folder = 'logs/'
        self.run = now.strftime('%Y%m%d-%H%M%S')
        #self.run = '20171013-212118'

        # directory definitions


        self.project_dir ='./'
        self.model_dir = self.project_dir + self.model_folder + self.mode + self.run
        self.logs_dir = self.project_dir + self.logs_folder + self.mode + self.run

        os.mkdir(self.model_dir)
        os.mkdir(self.logs_dir)

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

        for i in self.data_train:
            for j in i:
                
                self.imr_train.append(j['xcrLeft']/255.)                
                self.img_train.append(j['xcgLeft']/255.)
                self.img_train.append(j['xcgLeft']/255.)
                self.imb_train.append(j['xcbLeft']/255.)
                self.depth_train.append(j['xidLeft'])
                self.depth_mask_train.append(j['xmaskLeft'])
                self.gnd_train.append((j['semLeft']==1).astype(int))
                self.obj_train.append((j['semLeft']==2).astype(int))
                self.bld_train.append((j['semLeft']==3).astype(int))
                self.veg_train.append((j['semLeft']==4).astype(int))
                self.sky_train.append((j['semLeft']==5).astype(int))
              
                self.imr_train.append(j['xcrRight']/255.)
                self.img_train.append(j['xcgRight']/255.)
                self.imb_train.append(j['xcbRight']/255.)
                self.depth_train.append(j['xidRight'])
                self.depth_mask_train.append(j['xmaskRight'])
                self.gnd_train.append((j['semRight']==1).astype(int))
                self.obj_train.append((j['semRight']==2).astype(int))
                self.bld_train.append((j['semRight']==3).astype(int))
                self.veg_train.append((j['semRight']==4).astype(int))
                self.sky_train.append((j['semRight']==5).astype(int))
        
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

            for i in self.data_val:
                for j in i:
                    self.imr_val.append(j['xcrLeft']/255.)
                    self.img_val.append(j['xcgLeft']/255.)
                    self.imb_val.append(j['xcbLeft']/255.)
                    self.depth_val.append(j['xidLeft'])
                    self.gnd_val.append((j['semLeft']==1).astype(int))
                    self.obj_val.append((j['semLeft']==2).astype(int))
                    self.bld_val.append((j['semLeft']==3).astype(int))
                    self.veg_val.append((j['semLeft']==4).astype(int))
                    self.sky_val.append((j['semLeft']==5).astype(int))
                    self.depth_mask_val.append(j['xmaskLeft'])
            
                    self.imr_val.append(j['xcrRight']/255.)
                    self.img_val.append(j['xcgRight']/255.)
                    self.imb_val.append(j['xcbRight']/255.)
                    self.depth_val.append(j['xidRight'])
                    self.gnd_val.append((j['semRight']==1).astype(int))
                    self.obj_val.append((j['semRight']==2).astype(int))
                    self.bld_val.append((j['semRight']==3).astype(int))
                    self.veg_val.append((j['semRight']==4).astype(int))
                    self.sky_val.append((j['semRight']==5).astype(int))
                    self.depth_mask_val.append(j['xmaskRight'])
            
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

        sum_epoch_loss = tf.summary.scalar('Epoch_Loss_Red_Channel',epoch_loss)
        sum_val_loss = tf.summary.scalar('Validation_Loss_Red_Channel',val_loss)

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

        ind_batch = np.arange(0,self.batch_size)
        
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

            # comment for complete training on gpu
            #tf.get_default_graph().finalize()

            for epoch in range(0,self.hm_epochs):
                sess.run(epoch_loss_reset)
                time1 = datetime.datetime.now()

                for _ in range(0,n_batches):

                    imr_batch = self.imr_train[_*self.batch_size:(_+1)*self.batch_size,:]

                    imr_in = pretraining_input_distortion(copy(imr_batch))
                    
                    ind_rand_who = np.random.choice(ind_batch,self.batch_size/2,replace=False)
                    imr_in = horizontal_mirroring(imr_in,ind_rand_who)
                    imr_batch = horizontal_mirroring(imr_batch,ind_rand_who)
                    
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

        sum_epoch_loss = tf.summary.scalar('Epoch_Loss_Green_Channel',epoch_loss)
        sum_val_loss = tf.summary.scalar('Validation_Loss_Green_Channel',val_loss)

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

        ind_batch = np.arange(0,self.batch_size)
        
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

            #tf.get_default_graph().finalize()

            for epoch in range(0,self.hm_epochs):
                sess.run(epoch_loss_reset)
                time1 = datetime.datetime.now()

                for _ in range(0,n_batches):

                    img_batch = self.img_train[_*self.batch_size:(_+1)*self.batch_size,:]

                    img_in = pretraining_input_distortion(copy(img_batch))
                    
                    ind_rand_who = np.random.choice(ind_batch,self.batch_size/2,replace=False)
                    img_in = horizontal_mirroring(img_in,ind_rand_who)
                    img_batch = horizontal_mirroring(img_batch,ind_rand_who)
                    
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

        sum_epoch_loss = tf.summary.scalar('Epoch_Loss_Blue_Channel',epoch_loss)
        sum_val_loss = tf.summary.scalar('Validation_Loss_Blue_Channel',val_loss)

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

        ind_batch = np.arange(0,self.batch_size)
        
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

            #tf.get_default_graph().finalize()

            for epoch in range(0,self.hm_epochs):
                sess.run(epoch_loss_reset)
                time1 = datetime.datetime.now()

                for _ in range(0,n_batches):

                    imb_batch = self.imb_train[_*self.batch_size:(_+1)*self.batch_size,:]

                    imb_in = pretraining_input_distortion(copy(imb_batch))
                    
                    ind_rand_who = np.random.choice(ind_batch,self.batch_size/2,replace=False)
                    imb_in = horizontal_mirroring(imb_in,ind_rand_who)
                    imb_batch = horizontal_mirroring(imb_batch,ind_rand_who)
                    
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
        cost = tf.nn.l2_loss(tf.multiply(self.depth_mask,pred)-tf.multiply(self.depth_mask,self.label_depth))


        loss = tf.nn.l2_loss(tf.multiply(self.depth_mask,pred)-tf.multiply(self.depth_mask,self.label_depth))

        regularizer = tf.contrib.layers.l2_regularizer(scale=1e-05)
        reg = tf.contrib.layers.apply_regularization(regularizer,weights_list=[self.depth_ec_layer['weights'],
                                                                               self.depth_dc_layer['weights'],
                                                                               self.depth_ec_layer['bias'],
                                                                               self.depth_dc_layer['bias']])
        cost += reg

        epoch_loss = tf.Variable(0.0,name='epoch_loss',trainable=False)
        val_loss = tf.Variable(0.0,name='val_loss',trainable=False)

        sum_epoch_loss = tf.summary.scalar('Epoch_Loss_Depth_Channel',epoch_loss)
        sum_val_loss = tf.summary.scalar('Validation_Loss_Depth_Channel',val_loss)

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

        ind_batch = np.arange(0,self.batch_size)
        
        # validation objects
        validations = np.arange(0,self.n_validation_data)
        set_val = np.random.choice(validations,self.n_training_validations,replace=False)

        epoch_loss_reset = epoch_loss.assign(0)
        epoch_loss_update = epoch_loss.assign_add(cost)

        loss_val_reset = val_loss.assign(0)
        loss_val_update = val_loss.assign_add(loss)


        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.per_process_gpu_memory_fraction = 0.5

        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()

            train_writer1 = tf.summary.FileWriter(self.FLAGS.logs_dir,sess.graph)
            sess.run(tf.global_variables_initializer())

            n_batches = int(len(self.imb_train)/self.batch_size)

            #tf.get_default_graph().finalize()

            for epoch in range(0,self.hm_epochs):
                sess.run(epoch_loss_reset)
                time1 = datetime.datetime.now()

                for _ in range(0,n_batches):

                    batch = self.depth_train[_*self.batch_size:(_+1)*self.batch_size,:]
                    batch_mask = self.depth_mask_train[_*self.batch_size:(_+1)*self.batch_size,:]

                    depth_in = pretraining_input_distortion(copy(batch))

                    ind_rand_who = np.random.choice(ind_batch,self.batch_size/2,replace=False)
                    depth_in = horizontal_mirroring(depth_in,ind_rand_who)
                    batch = horizontal_mirroring(batch,ind_rand_who)
                    batch_mask = horizontal_mirroring(batch_mask,ind_rand_who)
                    
                    feed_dict = {self.input_depth:depth_in,
                                 self.label_depth:batch,
                                 self.depth_mask:batch_mask}

                    _, l = sess.run([opt, epoch_loss_update], feed_dict=feed_dict)


                sum_train = sess.run(sum_epoch_loss)
                train_writer1.add_summary(sum_train,epoch)

                print('----------------------------------------------------------------')
                print('Epoch', epoch, 'completed out of', self.hm_epochs)
                print('Training Loss (per epoch): ', sess.run(epoch_loss.value()))

                sess.run(loss_val_reset)

                normalization = 0
                for i in set_val:
                    depth_label = self.depth_val[i]
                    depth_mask = self.depth_mask_val[i]
                    normalization = normalization + np.count_nonzero(depth_mask)
                    depth_in = pretraining_input_distortion(copy(depth_label),singleframe=True)

                    feed_dict_val = {self.input_depth:depth_in,
                                     self.label_depth:[depth_label],
                                     self.depth_mask:[depth_mask]}

                    im_pred,c_val = sess.run([pred,loss_val_update],feed_dict=feed_dict_val)

                # uncomment when not running on gpu
                #if epoch%10==0:
                     #print_validation_frames(depth_in,im_pred,depth_label,channel='depth',shape=(60,18))

                sum_val = sess.run(sum_val_loss)
                train_writer1.add_summary(sum_val,epoch)
                print('Validation Loss (per pixel): ', sess.run(val_loss.value())/normalization)
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

        sum_epoch_loss = tf.summary.scalar('Epoch_Loss_Ground_Channel',epoch_loss)
        sum_val_loss = tf.summary.scalar('Validation_Loss_Ground_Channel',val_loss)

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

        ind_batch = np.arange(0,self.batch_size)
        
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
                    
                    ind_rand_who = np.random.choice(ind_batch,self.batch_size/2,replace=False)
                    inp = horizontal_mirroring(inp,ind_rand_who)
                    batch = horizontal_mirroring(batch,ind_rand_who)
                    
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

        sum_epoch_loss = tf.summary.scalar('Epoch_Loss_Object_Channel',epoch_loss)
        sum_val_loss = tf.summary.scalar('Validation_Loss_Object_Channel',val_loss)

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

        ind_batch = np.arange(0,self.batch_size)
        
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
                    
                    ind_rand_who = np.random.choice(ind_batch,self.batch_size/2,replace=False)
                    inp = horizontal_mirroring(inp,ind_rand_who)
                    batch = horizontal_mirroring(batch,ind_rand_who)

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

        sum_epoch_loss = tf.summary.scalar('Epoch_Loss_Building_Channel',epoch_loss)
        sum_val_loss = tf.summary.scalar('Validation_Loss_Building_Channel',val_loss)

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

        ind_batch = np.arange(0,self.batch_size)
        
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
                    
                    ind_rand_who = np.random.choice(ind_batch,self.batch_size/2,replace=False)
                    inp = horizontal_mirroring(inp,ind_rand_who)
                    batch = horizontal_mirroring(batch,ind_rand_who)

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

        sum_epoch_loss = tf.summary.scalar('Epoch_Loss_Vegetation_Channel',epoch_loss)
        sum_val_loss = tf.summary.scalar('Validation_Loss_Vegetation_Channel',val_loss)

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

        ind_batch = np.arange(0,self.batch_size)
        
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
                    
                    ind_rand_who = np.random.choice(ind_batch,self.batch_size/2,replace=False)
                    inp = horizontal_mirroring(inp,ind_rand_who)
                    batch = horizontal_mirroring(batch,ind_rand_who)

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

        sum_epoch_loss = tf.summary.scalar('Epoch_Loss_Sky_Channel',epoch_loss)
        sum_val_loss = tf.summary.scalar('Validation_Loss_Sky_Channel',val_loss)

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

        ind_batch = np.arange(0,self.batch_size)
        
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
                    
                    ind_rand_who = np.random.choice(ind_batch,self.batch_size/2,replace=False)
                    inp = horizontal_mirroring(inp,ind_rand_who)
                    batch = horizontal_mirroring(batch,ind_rand_who)

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

    def pretrain_shared_semantics(self,run = ''):

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

        sum_epoch_loss = tf.summary.scalar('Epoch_Loss_Shared_Semantics',epoch_loss)
        sum_val_loss = tf.summary.scalar('Validation_Loss_Shared_Semantics',val_loss)

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

        ind_batch = np.arange(0,self.batch_size)
        
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

        dir = 'models/pretraining/' + run

        with tf.Session(config=config) as sess:

            sess.run(tf.global_variables_initializer())
            saver_load_gnd.restore(sess,dir+'/pretrained_gnd.ckpt')
            saver_load_obj.restore(sess,dir+'/pretrained_obj.ckpt')
            saver_load_bld.restore(sess,dir+'/pretrained_bld.ckpt')
            saver_load_veg.restore(sess,dir+'/pretrained_veg.ckpt')
            saver_load_sky.restore(sess,dir+'/pretrained_sky.ckpt')

            train_writer1 = tf.summary.FileWriter(self.FLAGS.logs_dir,sess.graph)

            n_batches = int(len(self.gnd_train)/self.batch_size)

            #tf.get_default_graph().finalize()

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
                                                                                                        resolution=(18,60))
                    
                    ind_rand_who = np.random.choice(ind_batch,self.batch_size/2,replace=False)
                    
                    gnd_in = horizontal_mirroring(gnd_in,ind_rand_who)
                    gnd_batch = horizontal_mirroring(gnd_batch,ind_rand_who)
                    
                    obj_in = horizontal_mirroring(obj_in,ind_rand_who)
                    obj_batch = horizontal_mirroring(obj_batch,ind_rand_who)
                    
                    bld_in = horizontal_mirroring(bld_in,ind_rand_who)
                    bld_batch = horizontal_mirroring(bld_batch,ind_rand_who)
                    
                    veg_in = horizontal_mirroring(veg_in,ind_rand_who)
                    veg_batch = horizontal_mirroring(veg_batch,ind_rand_who)
                    
                    sky_in = horizontal_mirroring(sky_in,ind_rand_who)
                    sky_batch = horizontal_mirroring(sky_batch,ind_rand_who)

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

    def validate_depth(self,run=False):

        prediction = self.AE_depth(self.input_depth)
        loss = tf.nn.l2_loss(tf.multiply(self.depth_mask,prediction)-tf.multiply(self.depth_mask,self.label_depth))

        load_weights = tf.train.Saver()

        if run==False:
            raise ValueError

        dir = 'models/pretraining/' + run

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            load_weights.restore(sess,dir+'/pretrained_depth.ckpt')



            for i in range(0,self.n_validation_data):


                depth_label = self.depth_val[i]
                depth_mask = self.depth_mask_val[i]
                depth_input = pretraining_input_distortion(copy(depth_label),singleframe=True)

                normalization = np.count_nonzero(depth_mask)

                feed_dict = {self.input_depth:depth_input,
                             self.label_depth:[depth_label],
                             self.depth_mask:[depth_mask]}

                depth_pred, l = sess.run([prediction,loss],feed_dict=feed_dict)
                print_validation_frames(depth_input,depth_pred,depth_label,channel='depth',shape=(60,18))


                print('Validation Loss:', l/normalization)






#pretraining = PretrainingMAE(data_train, data_validate, data_test)

#pretraining.pretrain_red_channel()
#pretraining.pretrain_green_channel()
#pretraining.pretrain_blue_channel()

#pretraining.pretrain_gnd_channel()
#pretraining.pretrain_obj_channel()
#pretraining.pretrain_bld_channel()
#pretraining.pretrain_veg_channel()
#pretraining.pretrain_sky_channel()
#pretraining.pretrain_shared_semantics()
#pretraining.pretrain_depth_channel()

#pretraining.validate_depth(run='20171010-115125')
