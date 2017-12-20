# model based on 'Multi-modal Auto-Encoders as Joint Estimators for Robotics Scene Understanding' by Cadena et al.
# code developed by Silvan Weder


import tensorflow as tf
import numpy as np
import os
import evaluation_functions as eval
import basic_routines as BR

from input_distortion import input_distortion
from basic_routines import horizontal_mirroring, zeroing_channel
from datetime import datetime
from copy import copy, deepcopy
from models import full_MAE

from build_test_sequences import distort_test_sequences



class MAE:

    def __init__(self,n_epochs=None,learning_rate=None,mirroring=False,resolution=(18,60),verbose=False):

        if n_epochs == None:
            raise ValueError('no number of epochs passed')
        self.hm_epochs = n_epochs

        self.n_validations = 100

        if learning_rate == None:
            raise ValueError('no learning rate passed')
        self.learning_rate = learning_rate

        self.verbose = verbose

        self.mirroring = mirroring

        self.height = resolution[0]
        self.width = resolution[1]
        self.size_input = self.height*self.width

        self.size_coding = 1024

        # options
        self.mode = 'denoising' # standard for non denoising training or denoising for training a denoising MAE

        # noise values for the different modalities (min 0, max 1)
        self.imr_noise = 0.1
        self.img_noise = 0.1
        self.imb_noise = 0.1
        self.depth_noise = 0.1
        self.gnd_noise = 0.1
        self.obj_noise = 0.1
        self.bld_noise = 0.1
        self.veg_noise = 0.1
        self.sky_noise = 0.1

        # placeholder definition
        self.imr_input = tf.placeholder('float',[None,self.size_input])
        self.img_input = tf.placeholder('float',[None,self.size_input])
        self.imb_input = tf.placeholder('float',[None,self.size_input])
        self.depth_input = tf.placeholder('float',[None,self.size_input])
        self.gnd_input = tf.placeholder('float',[None,self.size_input])
        self.obj_input = tf.placeholder('float',[None,self.size_input])
        self.bld_input = tf.placeholder('float',[None,self.size_input])
        self.veg_input = tf.placeholder('float',[None,self.size_input])
        self.sky_input = tf.placeholder('float',[None,self.size_input])

        self.depth_mask = tf.placeholder('float',[None,self.size_input])

        self.imr_label = tf.placeholder('float',[None,self.size_input])
        self.img_label = tf.placeholder('float',[None,self.size_input])
        self.imb_label = tf.placeholder('float',[None,self.size_input])
        self.depth_label = tf.placeholder('float',[None,self.size_input])
        self.gnd_label = tf.placeholder('float',[None,self.size_input])
        self.obj_label = tf.placeholder('float',[None,self.size_input])
        self.bld_label = tf.placeholder('float',[None,self.size_input])
        self.veg_label = tf.placeholder('float',[None,self.size_input])
        self.sky_label = tf.placeholder('float',[None,self.size_input])

        # model savings
        self.saving = True
        now = datetime.now()

        self.folder_model = 'models/'
        self.folder_logs = 'logs/'

        self.mode = 'full/'
        self.run = now.strftime('%Y%m%d-%H%M%S')

        self.project_dir = './'
        self.model_dir = self.project_dir + self.folder_model + self.mode + self.run
        self.logs_dir = self.project_dir + self.folder_logs + self.mode + self.run

        os.mkdir(self.model_dir)
        os.mkdir(self.logs_dir)

        self.FLAGS = tf.app.flags.FLAGS

    def prepare_training_data(self,data_train):
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

        for i in data_train:
            for j in i:
                self.imr_train.append(j['xcr1']/255.)
                self.img_train.append(j['xcg1']/255.)
                self.imb_train.append(j['xcb1']/255.)
                self.depth_train.append(j['xid1'])
                self.depth_mask_train.append(j['xmask1'])
                self.gnd_train.append((j['sem1']==1).astype(int))
                self.obj_train.append((j['sem1']==2).astype(int))
                self.bld_train.append((j['sem1']==3).astype(int))
                self.veg_train.append((j['sem1']==4).astype(int))
                self.sky_train.append((j['sem1']==5).astype(int))

                self.imr_train.append(j['xcr2']/255.)
                self.img_train.append(j['xcg2']/255.)
                self.imb_train.append(j['xcb2']/255.)
                self.depth_train.append(j['xid2'])
                self.depth_mask_train.append(j['xmask2'])
                self.gnd_train.append((j['sem2']==1).astype(int))
                self.obj_train.append((j['sem2']==2).astype(int))
                self.bld_train.append((j['sem2']==3).astype(int))
                self.veg_train.append((j['sem2']==4).astype(int))
                self.sky_train.append((j['sem2']==5).astype(int))

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

    def prepare_validation_data(self,data_val):

        # prepare validation data containers
        self.imr_val = []
        self.img_val = []
        self.imb_val = []
        self.depth_val = []
        self.depth_mask_val = []
        self.gnd_val = []
        self.obj_val = []
        self.bld_val = []
        self.veg_val = []
        self.sky_val = []

        for i in data_val:
            for j in i:
                self.imr_val.append(j['xcr1']/255.)
                self.img_val.append(j['xcg1']/255.)
                self.imb_val.append(j['xcb1']/255.)
                self.depth_val.append(j['xid1'])
                self.depth_mask_val.append(j['xmask1'])
                self.gnd_val.append((j['sem1']==1).astype(int))
                self.obj_val.append((j['sem1']==2).astype(int))
                self.bld_val.append((j['sem1']==3).astype(int))
                self.veg_val.append((j['sem1']==4).astype(int))
                self.sky_val.append((j['sem1']==5).astype(int))

                self.imr_val.append(j['xcr2']/255.)
                self.img_val.append(j['xcg2']/255.)
                self.imb_val.append(j['xcb2']/255.)
                self.depth_val.append(j['xid2'])
                self.depth_mask_val.append(j['xmask2'])
                self.gnd_val.append((j['sem2']==1).astype(int))
                self.obj_val.append((j['sem2']==2).astype(int))
                self.bld_val.append((j['sem2']==3).astype(int))
                self.veg_val.append((j['sem2']==4).astype(int))
                self.sky_val.append((j['sem2']==5).astype(int))

        self.imr_val = np.asarray(self.imr_val)
        self.img_val = np.asarray(self.img_val)
        self.imb_val = np.asarray(self.imb_val)
        self.depth_val = np.asarray(self.depth_val)
        self.depth_mask_val = np.asarray(self.depth_mask_val)
        self.gnd_val = np.asarray(self.gnd_val)
        self.obj_val = np.asarray(self.obj_val)
        self.bld_val = np.asarray(self.bld_val)
        self.veg_val = np.asarray(self.veg_val)
        self.sky_val = np.asarray(self.sky_val)

    def prepare_test_data(self,data_test):

        # prepare test data containers
        self.imr_test = []
        self.img_test = []
        self.imb_test = []
        self.depth_test = []
        self.gnd_test = []
        self.obj_test = []
        self.bld_test = []
        self.veg_test = []
        self.sky_test = []

        for i in data_test:
            self.imr_test.append(i['xcr']/255.)
            self.img_test.append(i['xcg']/255.)
            self.imb_test.append(i['xcb']/255.)
            self.depth_test.append(i['xid'])
            self.gnd_test.append((i['sem']==1).astype(int))
            self.obj_test.append((i['sem']==2).astype(int))
            self.bld_test.append((i['sem']==3).astype(int))
            self.veg_test.append((i['sem']==4).astype(int))
            self.sky_test.append((i['sem']==5).astype(int))

    def prepare_test_frames(self,data_test):

        # prepare test data containers
        self.imr_test = []
        self.img_test = []
        self.imb_test = []
        self.depth_test = []
        self.gnd_test = []
        self.obj_test = []
        self.bld_test = []
        self.veg_test = []
        self.sky_test = []


        for j in data_test:
            self.imr_test.append(j['xcr']/255.)
            self.img_test.append(j['xcg']/255.)
            self.imb_test.append(j['xcb']/255.)
            self.depth_test.append(j['xid'])
            self.gnd_test.append((j['sem']==1).astype(int))
            self.obj_test.append((j['sem']==2).astype(int))
            self.bld_test.append((j['sem']==3).astype(int))
            self.veg_test.append((j['sem']==4).astype(int))
            self.sky_test.append((j['sem']==5).astype(int))

    def network(self,input):

        output = full_MAE(input[0],input[1],input[2],
                          input[3],input[4],input[5],
                          input[6],input[7],input[8])

        return output

    def train_model(self,data_train,data_validate,load='none',run=''):

        if self.verbose:
            print('[TRAINING]: prepare data')

        # prepare data
        self.prepare_training_data(data_train)
        self.prepare_validation_data(data_validate)

        # training options
        self.batch_size = 60
        self.n_batches = int(len(self.imr_train)/self.batch_size)
        self.hm_epoch_init = 20

        # print test flag
        if self.verbose:
            print('[TRAINING]: define model')

        # define input list
        input = [self.imr_input,self.img_input,self.imb_input,
                 self.depth_input,self.gnd_input,self.obj_input,
                 self.bld_input,self.veg_input,self.sky_input]

        # network call
        prediction = self.network(input)

        # regularizer initialization
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.005)
        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)

        # cost definition (training cost)
        cost = tf.nn.l2_loss(prediction[0]-self.imr_label) + \
               tf.nn.l2_loss(prediction[1]-self.img_label) + \
               tf.nn.l2_loss(prediction[2]-self.imb_label) + \
               0.1*tf.nn.l2_loss(prediction[4]-self.gnd_label) + \
               0.1*tf.nn.l2_loss(prediction[5]-self.obj_label) + \
               0.1*tf.nn.l2_loss(prediction[6]-self.bld_label) + \
               0.1*tf.nn.l2_loss(prediction[7]-self.veg_label) + \
               0.1*tf.nn.l2_loss(prediction[8]-self.sky_label) + \
               40*reg_term

        # depth mask for cost computation
        cost = cost + 50*tf.nn.l2_loss(tf.multiply(self.depth_mask,prediction[3])-tf.multiply(self.depth_mask,self.depth_label))
        # loss definition (validation loss)
        loss = tf.nn.l2_loss(prediction[0]-self.imr_label) + \
               tf.nn.l2_loss(prediction[1]-self.img_label) + \
               tf.nn.l2_loss(prediction[2]-self.imb_label) + \
               tf.nn.l2_loss(prediction[4]-self.gnd_label) + \
               tf.nn.l2_loss(prediction[5]-self.obj_label) + \
               tf.nn.l2_loss(prediction[6]-self.bld_label) + \
               tf.nn.l2_loss(prediction[7]-self.veg_label) + \
               tf.nn.l2_loss(prediction[8]-self.sky_label)

        # depth mask for loss computation
        loss = loss + tf.nn.l2_loss(tf.multiply(self.depth_mask,prediction[3])-tf.multiply(self.depth_mask,self.depth_label))

        # definition of all loss variables
        epoch_loss = tf.Variable(0.0,name='epoch_loss',trainable=False) # training cost
        val_loss = tf.Variable(0.0,name='val_loss',trainable=False) # validation loss
        imr_loss = tf.Variable(0.0,name='imr_val_loss',trainable=False) # validation loss of all separated channels
        img_loss = tf.Variable(0.0,name='img_val_loss',trainable=False)
        imb_loss = tf.Variable(0.0,name='imb_val_loss',trainable=False)
        gnd_loss = tf.Variable(0.0,name='gnd_val_loss',trainable=False)
        obj_loss = tf.Variable(0.0,name='obj_val_loss',trainable=False)
        bld_loss = tf.Variable(0.0,name='bld_val_loss',trainable=False)
        veg_loss = tf.Variable(0.0,name='veg_val_loss',trainable=False)
        sky_loss = tf.Variable(0.0,name='sky_val_loss',trainable=False)

        # validation loss of target metric
        rms = tf.Variable(0.0,name='rms_error',trainable=False)
        rel = tf.Variable(0.0,name='rel_error',trainable=False)

        # definition of normalization variable for losses
        normalization = tf.placeholder('float')

        # reset nodes for validation losses
        epoch_loss_reset = epoch_loss.assign(0)
        val_loss_reset = val_loss.assign(0)
        imr_loss_reset = imr_loss.assign(0.0)
        img_loss_reset = img_loss.assign(0.0)
        imb_loss_reset = imb_loss.assign(0.0)
        gnd_loss_reset = gnd_loss.assign(0.0)
        obj_loss_reset = obj_loss.assign(0.0)
        bld_loss_reset = bld_loss.assign(0.0)
        veg_loss_reset = veg_loss.assign(0.0)
        sky_loss_reset = sky_loss.assign(0.0)

        # update nodes for validation losses
        epoch_loss_update = epoch_loss.assign_add(cost)
        val_loss_update = val_loss.assign_add(loss/normalization)
        imr_loss_update = imr_loss.assign_add(tf.nn.l2_loss(self.imr_label-prediction[0])/normalization)
        img_loss_update = img_loss.assign_add(tf.nn.l2_loss(self.img_label-prediction[1])/normalization)
        imb_loss_update = imb_loss.assign_add(tf.nn.l2_loss(self.imb_label-prediction[2])/normalization)
        gnd_loss_update = gnd_loss.assign_add(tf.nn.l2_loss(self.gnd_label-prediction[4])/normalization)
        obj_loss_update = obj_loss.assign_add(tf.nn.l2_loss(self.obj_label-prediction[5])/normalization)
        bld_loss_update = bld_loss.assign_add(tf.nn.l2_loss(self.bld_label-prediction[6])/normalization)
        veg_loss_update = veg_loss.assign_add(tf.nn.l2_loss(self.veg_label-prediction[7])/normalization)
        sky_loss_update = sky_loss.assign_add(tf.nn.l2_loss(self.sky_label-prediction[8])/normalization)

        # target metric placeholder
        rms_plh = tf.placeholder('float')
        rel_plh = tf.placeholder('float')

        # target metric reset
        rms_reset = rms.assign(0.0)
        rel_reset = rel.assign(0.0)

        # target metric update
        rms_update = rms.assign_add(rms_plh)
        rel_update = rel.assign_add(rel_plh)

        # summary definitions
        sum_epoch_loss = tf.summary.scalar('Epoch_Loss_Full_Model',epoch_loss)
        sum_val_loss = tf.summary.scalar('Validation_Loss_Full_Model',val_loss)
        summary_imr = tf.summary.scalar('Red Channel Validation Loss', imr_loss)
        summary_img = tf.summary.scalar('Green Channel Validation Loss', img_loss)
        summary_imb = tf.summary.scalar('Blue Channel Validation Loss', imb_loss)
        summary_gnd = tf.summary.scalar('Ground Channel Validation Loss', gnd_loss)
        summary_obj = tf.summary.scalar('Object Channel Validation Loss', obj_loss)
        summary_bld = tf.summary.scalar('Building Channel Validation Loss', bld_loss)
        summary_veg = tf.summary.scalar('Vegetation Channel Validation Loss', veg_loss)
        summary_sky = tf.summary.scalar('Sky Channel Validation Loss', sky_loss)
        summary_rms = tf.summary.scalar('RMS Error in Validation', rms)
        summary_rel = tf.summary.scalar('Relative Error in Validation', rel)


        # learning rate defintion and options
        global_step = tf.Variable(0,trainable=False)
        base_rate = self.learning_rate
        self.learning_rate = tf.train.exponential_decay(base_rate,global_step,100000, 0.96, staircase=True)

        # variable list for training
        var_list = []

        # get graph variables
        with tf.variable_scope("Encoding",reuse=True):
            var_list.append(tf.get_variable('full_ec_layer_weights'))
            var_list.append(tf.get_variable('full_ec_layer_bias'))

        with tf.variable_scope("Decoding",reuse=True):
            var_list.append(tf.get_variable('full_dc_layer_weights'))
            var_list.append(tf.get_variable('full_dc_layer_bias'))

        # optimizer definition
        with tf.name_scope('Optimizer'):
            optimizer0 = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost,global_step=global_step)
            optimizer1 = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost,var_list=var_list,global_step=global_step)
            optimizer2 = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost,global_step=global_step)
            optimizer3 = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost,global_step=global_step)

        # batch indices
        ind_batch = np.arange(0,self.batch_size)

        # validation indices
        validations = np.arange(0, len(self.imr_val))

        # validation set
        set_val = np.random.choice(validations,self.n_validations,replace=False)
        set_val = list(set_val)

        # minimum validation loss initialization
        val_loss_min = np.infty
        save_count = 0

        # verbose?
        if self.verbose:
            print('[TRAINING]: load pretrained models')

        # define loaders
        if load == 'pretrained':

            with tf.variable_scope("Encoding",reuse=True):
                load_imr_ec = tf.train.Saver({'red_ec_layer_weights':tf.get_variable('red_ec_layer_weights'),
                                              'red_ec_layer_bias':tf.get_variable('red_ec_layer_bias')})

                load_img_ec = tf.train.Saver({'green_ec_layer_weights':tf.get_variable('green_ec_layer_weights'),
                                              'green_ec_layer_bias':tf.get_variable('green_ec_layer_bias')})

                load_imb_ec = tf.train.Saver({'blue_ec_layer_weights':tf.get_variable('blue_ec_layer_weights'),
                                              'blue_ec_layer_bias':tf.get_variable('blue_ec_layer_bias')})

                load_dpt_ec = tf.train.Saver({'depth_ec_layer_weights':tf.get_variable('depth_ec_layer_weights'),
                                              'depth_ec_layer_bias':tf.get_variable('depth_ec_layer_bias')})

                load_gnd_ec = tf.train.Saver({'gnd_ec_layer_weights':tf.get_variable('gnd_ec_layer_weights'),
                                              'gnd_ec_layer_bias':tf.get_variable('gnd_ec_layer_bias')})

                load_obj_ec = tf.train.Saver({'obj_ec_layer_weights':tf.get_variable('obj_ec_layer_weights'),
                                              'obj_ec_layer_bias':tf.get_variable('obj_ec_layer_bias')})

                load_bld_ec = tf.train.Saver({'bld_ec_layer_weights':tf.get_variable('bld_ec_layer_weights'),
                                              'bld_ec_layer_bias':tf.get_variable('bld_ec_layer_bias')})

                load_veg_ec = tf.train.Saver({'veg_ec_layer_weights':tf.get_variable('veg_ec_layer_weights'),
                                              'veg_ec_layer_bias':tf.get_variable('veg_ec_layer_bias')})

                load_sky_ec = tf.train.Saver({'sky_ec_layer_weights':tf.get_variable('sky_ec_layer_weights'),
                                              'sky_ec_layer_bias':tf.get_variable('sky_ec_layer_bias')})

                load_sem_ec = tf.train.Saver({'sem_ec_layer_weights':tf.get_variable('sem_ec_layer_weights'),
                                              'sem_ec_layer_bias':tf.get_variable('sem_ec_layer_bias')})

                load_shd_ec = tf.train.Saver({'full_ec_layer_weights':tf.get_variable('sky_ec_layer_weights'),
                                              'full_ec_layer_bias':tf.get_variable('sky_ec_layer_bias')})

            with tf.variable_scope("Decoding",reuse=True):
                load_imr_dc = tf.train.Saver({'red_dc_layer_weights':tf.get_variable('red_dc_layer_weights'),
                                              'red_dc_layer_bias':tf.get_variable('red_dc_layer_bias')})

                load_img_dc = tf.train.Saver({'green_dc_layer_weights':tf.get_variable('green_dc_layer_weights'),
                                              'green_dc_layer_bias':tf.get_variable('green_dc_layer_bias')})

                load_imb_dc = tf.train.Saver({'blue_dc_layer_weights':tf.get_variable('blue_dc_layer_weights'),
                                              'blue_dc_layer_bias':tf.get_variable('blue_dc_layer_bias')})

                load_dpt_dc = tf.train.Saver({'depth_dc_layer_weights':tf.get_variable('depth_dc_layer_weights'),
                                              'depth_dc_layer_bias':tf.get_variable('depth_dc_layer_bias')})

                load_gnd_dc = tf.train.Saver({'gnd_dc_layer_weights':tf.get_variable('gnd_dc_layer_weights'),
                                              'gnd_dc_layer_bias':tf.get_variable('gnd_dc_layer_bias')})

                load_obj_dc = tf.train.Saver({'obj_dc_layer_weights':tf.get_variable('obj_dc_layer_weights'),
                                              'obj_dc_layer_bias':tf.get_variable('obj_dc_layer_bias')})

                load_bld_dc = tf.train.Saver({'bld_dc_layer_weights':tf.get_variable('bld_dc_layer_weights'),
                                              'bld_dc_layer_bias':tf.get_variable('bld_dc_layer_bias')})

                load_veg_dc = tf.train.Saver({'veg_dc_layer_weights':tf.get_variable('veg_dc_layer_weights'),
                                              'veg_dc_layer_bias':tf.get_variable('veg_dc_layer_bias')})

                load_sky_dc = tf.train.Saver({'sky_dc_layer_weights':tf.get_variable('sky_dc_layer_weights'),
                                              'sky_dc_layer_bias':tf.get_variable('sky_dc_layer_bias')})

                load_sem_dc = tf.train.Saver({'sem_dc_layer_weights':tf.get_variable('sem_dc_layer_weights'),
                                              'sem_dc_layer_bias':tf.get_variable('sem_dc_layer_bias')})

                load_shd_dc = tf.train.Saver({'full_dc_layer_weights':tf.get_variable('sky_dc_layer_weights'),
                                              'full_dc_layer_bias':tf.get_variable('sky_dc_layer_bias')})

            saver = tf.train.Saver()

        # define loaders
        if load == 'continue':

            with tf.variable_scope('Encoding',reuse=True):
                load_ec_full = tf.train.Saver({'red_ec_layer_weights':tf.get_variable('red_ec_layer_weights'),
                                               'red_ec_layer_bias':tf.get_variable('red_ec_layer_bias'),
                                               'green_ec_layer_weights':tf.get_variable('green_ec_layer_weights'),
                                               'green_ec_layer_bias':tf.get_variable('green_ec_layer_bias'),
                                               'blue_ec_layer_weights':tf.get_variable('blue_ec_layer_weights'),
                                               'blue_ec_layer_bias':tf.get_variable('blue_ec_layer_bias'),
                                               'depth_ec_layer_weights':tf.get_variable('depth_ec_layer_weights'),
                                               'depth_ec_layer_bias':tf.get_variable('depth_ec_layer_bias'),
                                               'gnd_ec_layer_weights':tf.get_variable('gnd_ec_layer_weights'),
                                               'gnd_ec_layer_bias':tf.get_variable('gnd_ec_layer_bias'),
                                               'obj_ec_layer_weights':tf.get_variable('obj_ec_layer_weights'),
                                               'obj_ec_layer_bias':tf.get_variable('obj_ec_layer_bias'),
                                               'bld_ec_layer_weights':tf.get_variable('bld_ec_layer_weights'),
                                               'bld_ec_layer_bias':tf.get_variable('bld_ec_layer_bias'),
                                               'veg_ec_layer_weights':tf.get_variable('veg_ec_layer_weights'),
                                               'veg_ec_layer_bias':tf.get_variable('veg_ec_layer_bias'),
                                               'sky_ec_layer_weights':tf.get_variable('sky_ec_layer_weights'),
                                               'sky_ec_layer_bias':tf.get_variable('sky_ec_layer_bias'),
                                               'sem_ec_layer_weights':tf.get_variable('sem_ec_layer_weights'),
                                               'sem_ec_layer_bias':tf.get_variable('sem_ec_layer_bias'),
                                               'full_ec_layer_weights':tf.get_variable('full_ec_layer_weights'),
                                               'full_ec_layer_bias':tf.get_variable('full_ec_layer_bias')})

            with tf.variable_scope('Decoding',reuse=True):
                load_dc_full = tf.train.Saver({'red_dc_layer_weights':tf.get_variable('red_dc_layer_weights'),
                                               'red_dc_layer_bias':tf.get_variable('red_dc_layer_bias'),
                                               'green_dc_layer_weights':tf.get_variable('green_dc_layer_weights'),
                                               'green_dc_layer_bias':tf.get_variable('green_dc_layer_bias'),
                                               'blue_dc_layer_weights':tf.get_variable('blue_dc_layer_weights'),
                                               'blue_dc_layer_bias':tf.get_variable('blue_dc_layer_bias'),
                                               'depth_dc_layer_weights':tf.get_variable('depth_dc_layer_weights'),
                                               'depth_dc_layer_bias':tf.get_variable('depth_dc_layer_bias'),
                                               'gnd_dc_layer_weights':tf.get_variable('gnd_dc_layer_weights'),
                                               'gnd_dc_layer_bias':tf.get_variable('gnd_dc_layer_bias'),
                                               'obj_dc_layer_weights':tf.get_variable('obj_dc_layer_weights'),
                                               'obj_dc_layer_bias':tf.get_variable('obj_dc_layer_bias'),
                                               'bld_dc_layer_weights':tf.get_variable('bld_dc_layer_weights'),
                                               'bld_dc_layer_bias':tf.get_variable('bld_dc_layer_bias'),
                                               'veg_dc_layer_weights':tf.get_variable('veg_dc_layer_weights'),
                                               'veg_dc_layer_bias':tf.get_variable('veg_dc_layer_bias'),
                                               'sky_dc_layer_weights':tf.get_variable('sky_dc_layer_weights'),
                                               'sky_dc_layer_bias':tf.get_variable('sky_dc_layer_bias'),
                                               'sem_dc_layer_weights':tf.get_variable('sem_dc_layer_weights'),
                                               'sem_dc_layer_bias':tf.get_variable('sem_dc_layer_bias'),
                                               'full_dc_layer_weights':tf.get_variable('full_dc_layer_weights'),
                                               'full_dc_layer_bias':tf.get_variable('full_dc_layer_bias')})

        # saver initialization
        saver = tf.train.Saver()


        if self.verbose:
            print('[TRAINING]: start session')

        # session configuration
        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.per_process_gpu_memory_fraction = 0.5

        # start session
        with tf.Session(config=config) as sess:

            if self.verbose:
                print('[TRAINING]: start session - done')

            # writer initialization
            train_writer1 = tf.summary.FileWriter(self.logs_dir,sess.graph)

            # initialize all variables
            sess.run(tf.global_variables_initializer())

            if self.verbose:
                print('[TRAINING]: restore variables')

            # load variables
            if load == 'pretrained':
                folder = 'models/pretraining/pretrained_models/' + run
                load_imr_ec.restore(sess, folder + 'pretrained_red.ckpt')
                load_imr_dc.restore(sess,folder + 'pretrained_red.ckpt')
                load_img_ec.restore(sess, folder + 'pretrained_green.ckpt')
                load_img_dc.restore(sess,folder + 'pretrained_green.ckpt')
                load_imb_ec.restore(sess, folder + 'pretrained_blue.ckpt')
                load_imb_dc.restore(sess,folder + 'pretrained_blue.ckpt')
                load_dpt_ec.restore(sess, folder + 'pretrained_depth.ckpt')
                load_dpt_dc.restore(sess,folder + 'pretrained_depth.ckpt')
                load_gnd_ec.restore(sess,folder + 'pretrained_shared_semantics.ckpt')
                load_gnd_dc.restore(sess,folder + 'pretrained_shared_semantics.ckpt')
                load_obj_ec.restore(sess,folder + 'pretrained_shared_semantics.ckpt')
                load_obj_dc.restore(sess,folder + 'pretrained_shared_semantics.ckpt')
                load_bld_ec.restore(sess,folder + 'pretrained_shared_semantics.ckpt')
                load_bld_dc.restore(sess,folder + 'pretrained_shared_semantics.ckpt')
                load_veg_ec.restore(sess,folder + 'pretrained_shared_semantics.ckpt')
                load_veg_dc.restore(sess,folder + 'pretrained_shared_semantics.ckpt')
                load_sky_ec.restore(sess,folder + 'pretrained_shared_semantics.ckpt')
                load_sky_dc.restore(sess,folder + 'pretrained_shared_semantics.ckpt')

            if load == 'continue':
                folder = 'models/full/' + run + '/'
                load_ec_full.restore(sess, folder + 'fullmodel.ckpt')
                load_dc_full.restore(sess, folder + 'fullmodel.ckpt')

            # finalize graph
            tf.get_default_graph().finalize()

            if self.verbose:
                print('[TRAINING]: start training epochs')

            # start training epochs
            for epoch in range(0,self.hm_epochs):

                sess.run(epoch_loss_reset)
                time1 = datetime.now()

                for _ in range(self.n_batches):

                    imr_batch = self.imr_train[_*self.batch_size:(_+1)*self.batch_size]
                    img_batch = self.img_train[_*self.batch_size:(_+1)*self.batch_size]
                    imb_batch = self.imb_train[_*self.batch_size:(_+1)*self.batch_size]
                    depth_batch = self.depth_train[_*self.batch_size:(_+1)*self.batch_size]
                    gnd_batch = self.gnd_train[_*self.batch_size:(_+1)*self.batch_size]
                    obj_batch = self.obj_train[_*self.batch_size:(_+1)*self.batch_size]
                    bld_batch = self.bld_train[_*self.batch_size:(_+1)*self.batch_size]
                    veg_batch = self.veg_train[_*self.batch_size:(_+1)*self.batch_size]
                    sky_batch = self.sky_train[_*self.batch_size:(_+1)*self.batch_size]

                    depth_mask_batch = self.depth_mask_train[_*self.batch_size:(_+1)*self.batch_size]

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

                    # horizontal mirroring
                    if self.mirroring:

                        ind_rand_who = np.random.choice(ind_batch,int(self.batch_size/2),replace=False)

                        imr_in = horizontal_mirroring(imr_in,ind_rand_who)
                        imr_batch = horizontal_mirroring(imr_batch,ind_rand_who)
                        img_in = horizontal_mirroring(img_in,ind_rand_who)
                        img_batch = horizontal_mirroring(img_batch,ind_rand_who)
                        imb_in = horizontal_mirroring(imb_in,ind_rand_who)
                        imb_batch = horizontal_mirroring(imb_batch,ind_rand_who)

                        depth_in = horizontal_mirroring(depth_in,ind_rand_who)
                        depth_batch = horizontal_mirroring(depth_batch,ind_rand_who)
                        depth_mask_batch = horizontal_mirroring(depth_mask_batch,ind_rand_who)

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

                    # zeroing depth inputs
                    ind_rand_who = np.random.choice(ind_batch,int(self.batch_size/2),replace=False)

                    depth_in = zeroing_channel(depth_in,ind_rand_who)

                    # zeroing semantic inputs
                    ind_rand_who = np.random.choice(ind_batch,int(self.batch_size/2),replace=False)

                    gnd_in = zeroing_channel(gnd_in,ind_rand_who)
                    obj_in = zeroing_channel(obj_in,ind_rand_who)
                    bld_in = zeroing_channel(bld_in,ind_rand_who)
                    veg_in = zeroing_channel(veg_in,ind_rand_who)
                    sky_in = zeroing_channel(sky_in,ind_rand_who)


                    feed_dict = {self.imr_input:imr_in,
                                 self.img_input:img_in,
                                 self.imb_input:imb_in,
                                 self.depth_input:depth_in,
                                 self.gnd_input:gnd_in,
                                 self.obj_input:obj_in,
                                 self.bld_input:bld_in,
                                 self.veg_input:veg_in,
                                 self.sky_input:sky_in,
                                 self.depth_mask:depth_mask_batch,
                                 self.imr_label:imr_batch,
                                 self.img_label:img_batch,
                                 self.imb_label:imb_batch,
                                 self.depth_label:depth_batch,
                                 self.gnd_label:gnd_batch,
                                 self.obj_label:obj_batch,
                                 self.bld_label:bld_batch,
                                 self.veg_label:veg_batch,
                                 self.sky_label:sky_batch}

                    if load == 'none':
                        _ , c, l = sess.run([optimizer0, cost, epoch_loss_update], feed_dict=feed_dict)
                    if load == 'pretrained':
                        # training operation (first only full encoding is trained, then (after 10 epochs) everything is trained
                        if epoch < self.hm_epoch_init:
                            _ , c, l = sess.run([optimizer1, cost, epoch_loss_update], feed_dict=feed_dict)
                        else:
                            _ ,c, l = sess.run([optimizer2, cost, epoch_loss_update],feed_dict=feed_dict)
                    if load == 'continue':
                        _ , c, l = sess.run([optimizer3, cost, epoch_loss_update], feed_dict=feed_dict)

                sum_train = sess.run(sum_epoch_loss)
                train_writer1.add_summary(sum_train,epoch)

                if self.verbose:
                    print('----------------------------------------------------------------')
                    print('Epoch', epoch, 'completed out of', self.hm_epochs)
                    print('Training Loss (per epoch): ', sess.run(epoch_loss.value()))



                sess.run([val_loss_reset,rms_reset,rel_reset,
                          imr_loss_reset,img_loss_reset,imb_loss_reset,
                          gnd_loss_reset,obj_loss_reset,bld_loss_reset,veg_loss_reset,sky_loss_reset])

                norm = 8.*1080*len(set_val)

                error_rms = 0
                error_rel = 0


                red_label = self.imr_val[set_val]
                green_label = self.img_val[set_val]
                blue_label = self.imb_val[set_val]
                depth_label = self.depth_val[set_val]
                depth_mask = self.depth_mask_val[set_val]
                gnd_label = self.gnd_val[set_val]
                obj_label = self.obj_val[set_val]
                bld_label = self.bld_val[set_val]
                veg_label = self.veg_val[set_val]
                sky_label = self.sky_val[set_val]

                norm += np.count_nonzero(depth_mask)



                imr_in,img_in,imb_in,depth_in,gnd_in,obj_in,bld_in,veg_in,sky_in = input_distortion(copy(red_label),
                                                                                                    copy(green_label),
                                                                                                    copy(blue_label),
                                                                                                    copy(depth_label),
                                                                                                    copy(gnd_label),
                                                                                                    copy(obj_label),
                                                                                                    copy(bld_label),
                                                                                                    copy(veg_label),
                                                                                                    copy(sky_label),
                                                                                                    resolution=(18,60),
                                                                                                    singleframe=False)

                feed_dict = {self.imr_input:imr_in,
                                 self.img_input:img_in,
                                 self.imb_input:imb_in,
                                 self.depth_input:depth_in,
                                 self.gnd_input:gnd_in,
                                 self.obj_input:obj_in,
                                 self.bld_input:bld_in,
                                 self.veg_input:veg_in,
                                 self.sky_input:sky_in,
                                 self.depth_mask:depth_mask,
                                 self.imr_label:red_label,
                                 self.img_label:green_label,
                                 self.imb_label:blue_label,
                                 self.depth_label:depth_label,
                                 self.gnd_label:gnd_label,
                                 self.obj_label:obj_label,
                                 self.bld_label:bld_label,
                                 self.veg_label:veg_label,
                                 self.sky_label:sky_label,
                                 normalization:norm}

                im_pred,c_val,c_imr,c_img,c_imb,c_gnd,c_obj, c_bld, c_veg, c_sky = sess.run([prediction,
                                                                                             val_loss_update,
                                                                                             imr_loss_update,
                                                                                             img_loss_update,
                                                                                             imb_loss_update,
                                                                                             gnd_loss_update,
                                                                                             obj_loss_update,
                                                                                             bld_loss_update,
                                                                                             veg_loss_update,
                                                                                             sky_loss_update],
                                                                                             feed_dict=feed_dict)
                for frame in range(0,self.n_validations):
                    depth_pred = BR.invert_depth(im_pred[3][frame,:])
                    depth_gt = BR.invert_depth(depth_label[frame])

                    error_rms += eval.rms_error(depth_pred,depth_gt)
                    error_rel += eval.relative_error(depth_pred,depth_gt)

                timeb = datetime.now()


                error_rms = error_rms/len(set_val)
                error_rel = error_rel/len(set_val)

                sess.run(rms_update,feed_dict={rms_plh:error_rms})
                sess.run(rel_update,feed_dict={rel_plh:error_rel})

                sum_val = sess.run(sum_val_loss)
                sum_rms = sess.run(summary_rms)
                sum_rel = sess.run(summary_rel)
                sum_imr = sess.run(summary_imr)
                sum_img = sess.run(summary_img)
                sum_imb = sess.run(summary_imb)
                sum_gnd = sess.run(summary_gnd)
                sum_obj = sess.run(summary_obj)
                sum_bld = sess.run(summary_bld)
                sum_veg = sess.run(summary_veg)
                sum_sky = sess.run(summary_sky)

                train_writer1.add_summary(sum_val,epoch)
                train_writer1.add_summary(sum_rms,epoch)
                train_writer1.add_summary(sum_rel,epoch)
                train_writer1.add_summary(sum_imr,epoch)
                train_writer1.add_summary(sum_img,epoch)
                train_writer1.add_summary(sum_imb,epoch)
                train_writer1.add_summary(sum_gnd,epoch)
                train_writer1.add_summary(sum_obj,epoch)
                train_writer1.add_summary(sum_bld,epoch)
                train_writer1.add_summary(sum_veg,epoch)
                train_writer1.add_summary(sum_sky,epoch)

                if self.verbose:
                    print('Validation Loss (per pixel): ', sess.run(val_loss.value()))
                    time2 = datetime.now()
                    delta = time2-time1
                    print('Epoch Time [seconds]:', delta.seconds)
                    print('-----------------------------------------------------------------')

                if epoch%5 == 0:
                    v = sess.run(val_loss.value())
                    if v < val_loss_min:
                        val_loss_min = v
                        saver.save(sess,self.model_dir+'/fullmodel.ckpt')
                        save_count = 0
                    else:
                        save_count += 1
                        if save_count > 2:
                            self.learning_rate *= 1e-1

                        if save_count > 10:
                            break

    def validate_model(self,n_validations,run,loadmodel=True):

        with tf.Session() as sess:

            prediction = self.neural_model(self.imr_input,
                                       self.img_input,
                                       self.imb_input,
                                       self.depth_input,
                                       self.gnd_input,
                                       self.obj_input,
                                       self.bld_input,
                                       self.veg_input,
                                       self.sky_input,
                                       mode='training')

            #init_op = tf.initialize_all_variables()
            saver = tf.train.Saver()
            if loadmodel == True:
                dir = self.project_dir + self.folder_model + self.mode + run + 'fullmodel.ckpt'
                saver.restore(sess,dir)

            #sess.run(init_op)

            for i in range(0,n_validations):
                imr_out = self.imr_val[i]
                img_out = self.img_val[i]
                imb_out = self.imb_val[i]
                depth_out = self.depth_val[i]
                gnd_out = self.gnd_val[i]
                obj_out = self.obj_val[i]
                bld_out = self.bld_val[i]
                veg_out = self.veg_val[i]
                sky_out = self.sky_val[i]


                imr_in,img_in,imb_in,depth_in,gnd_in,obj_in,bld_in,veg_in,sky_in = input_distortion(imr_out,
                                                                                                    img_out,
                                                                                                    imb_out,
                                                                                                    depth_out,
                                                                                                    gnd_out,
                                                                                                    obj_out,
                                                                                                    bld_out,
                                                                                                    veg_out,
                                                                                                    sky_out,
                                                                                                    resolution=(18,60),
                                                                                                    singleframe=True)


                feed_dict = {self.imr_input:imr_in,
                             self.img_input:img_in,
                             self.imb_input:imb_in,
                             self.depth_input:depth_in,
                             self.gnd_input:gnd_in,
                             self.obj_input:obj_in,
                             self.bld_input:bld_in,
                             self.veg_input:veg_in,
                             self.sky_input:sky_in}

                prediction = sess.run(prediction,feed_dict=feed_dict)

    def evaluate(self,data_test,run=False):

        self.prepare_test_data(data_test)

        predictions = self.neural_model(self.imr_input,
                                        self.img_input,
                                        self.imb_input,
                                        self.depth_input,
                                        self.gnd_input,
                                        self.obj_input,
                                        self.bld_input,
                                        self.veg_input,
                                        self.sky_input)

        load_weights = tf.train.Saver()

        if run==False:
            raise ValueError

        dir = 'models/full/' + run

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            load_weights.restore(sess,dir+'/fullmodel.ckpt')

            n_evaluations = len(self.imr_test)
            print('Size of Test Set:',n_evaluations)

            error_rms = 0
            error_rel = 0

            all_inv_depth_pred = np.empty([1,0])
            all_inv_depth_label = np.empty([1,0])

            for i in range(0,n_evaluations):

                imr_label = self.imr_test[i]
                img_label = self.img_test[i]
                imb_label = self.imb_test[i]
                depth_label = self.depth_test[i]
                gnd_label = self.gnd_test[i]
                obj_label = self.obj_test[i]
                bld_label = self.bld_test[i]
                veg_label = self.veg_test[i]
                sky_label = self.sky_test[i]

                # taking only rgb as input
                depth_in = [[0]*self.size_input]
                #gnd_in = [[0]*self.size_input]
                #obj_in = [[0]*self.size_input]
                #bld_in = [[0]*self.size_input]
                #veg_in = [[0]*self.size_input]
                #sky_in = [[0]*self.size_input]

                gnd_in = [gnd_label]
                obj_in = [obj_label]
                bld_in = [bld_label]
                veg_in = [veg_label]
                sky_in = [sky_label]



                feed_dict = {self.imr_input:[imr_label],
                             self.img_input:[img_label],
                             self.imb_input:[imb_label],
                             self.depth_input:depth_in,
                             self.gnd_input:gnd_in,
                             self.obj_input:obj_in,
                             self.bld_input:bld_in,
                             self.veg_input:veg_in,
                             self.sky_input:sky_in}

                pred = sess.run(predictions,feed_dict=feed_dict)
                depth_pred = pred[3]


                all_inv_depth_pred = np.concatenate((all_inv_depth_pred,np.asarray(depth_pred)),axis=1)
                all_inv_depth_label = np.concatenate((all_inv_depth_label,np.asarray([depth_label])),axis=1)

            gt = BR.invert_depth(all_inv_depth_label)
            est = BR.invert_depth(all_inv_depth_pred)

            error_rms = eval.rms_error(est,gt)
            error_rel = eval.relative_error(est,gt)


            print('Error (RMS):', error_rms)
            print('Error (Relative Error):', error_rel)

    def evaluate_per_frame(self,data_test,run=False):

        self.prepare_test_frames(data_test)

        predictions = self.neural_model(self.imr_input,
                                        self.img_input,
                                        self.imb_input,
                                        self.depth_input,
                                        self.gnd_input,
                                        self.obj_input,
                                        self.bld_input,
                                        self.veg_input,
                                        self.sky_input)

        load_weights = tf.train.Saver()

        if run==False:
            raise ValueError

        dir = 'models/full/' + run

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            load_weights.restore(sess,dir+'/fullmodel.ckpt')

            n_evaluations = len(self.imr_test)
            print('Size of Test Set:',n_evaluations)

            error_rms = 0
            error_rel = 0

            all_inv_depth_pred = np.empty([1,0])
            all_inv_depth_label = np.empty([1,0])

            all_sem_pred = np.empty([5,0,len(self.gnd_test[0])])
            all_sem_label = np.empty([5,0,len(self.gnd_test[0])])

            for i in range(0,n_evaluations):

                imr_label = self.imr_test[i]
                img_label = self.img_test[i]
                imb_label = self.imb_test[i]
                depth_label = self.depth_test[i]
                gnd_label = self.gnd_test[i]
                obj_label = self.obj_test[i]
                bld_label = self.bld_test[i]
                veg_label = self.veg_test[i]
                sky_label = self.sky_test[i]


                # taking only rgb as input
                depth_in = [[0]*self.size_input]
                gnd_in = [[0]*self.size_input]
                obj_in = [[0]*self.size_input]
                bld_in = [[0]*self.size_input]
                veg_in = [[0]*self.size_input]
                sky_in = [[0]*self.size_input]

                feed_dict = {self.imr_input:[imr_label],
                             self.img_input:[img_label],
                             self.imb_input:[imb_label],
                             self.depth_input:depth_in,
                             self.gnd_input:gnd_in,
                             self.obj_input:obj_in,
                             self.bld_input:bld_in,
                             self.veg_input:veg_in,
                             self.sky_input:sky_in}

                pred = sess.run(predictions,feed_dict=feed_dict)
                depth_pred = pred[3]

                all_inv_depth_pred = np.concatenate((all_inv_depth_pred,np.asarray(depth_pred)),axis=1)
                all_inv_depth_label = np.concatenate((all_inv_depth_label,np.asarray([depth_label])),axis=1)

                all_sem_pred = np.concatenate((all_sem_pred,np.asarray(pred[4:])),axis=1)

                sem_label = np.concatenate((np.asarray([gnd_label]),np.asarray([obj_label]),np.asarray([bld_label]),
                                           np.asarray([veg_label]),np.asarray([sky_label])),axis=0)

                all_sem_label = np.concatenate((all_sem_label,sem_label[:,np.newaxis,:]),axis=1)

            depth_gt = BR.invert_depth(all_inv_depth_label)
            depth_est = BR.invert_depth(all_inv_depth_pred)

            error_rms = eval.rms_error(depth_est,depth_gt)
            error_rel = eval.relative_error(depth_est,depth_gt)

            iu_semantics, inter, union = eval.inter_union(all_sem_pred,all_sem_label)

            print('Error (RMS):', error_rms)
            print('Error (Relative Error):', error_rel)

            print('Intersection over Union per class:')
            print(iu_semantics)

            print('Intersection: ', inter)
            print('Union: ', union)

    '''
    def forward_rgb(self,x,run=False):
        
        # x is an image of shape w x h x 3 in [0,1]         
        # prepare data containers        
        imr_in = x[:][:][0]
        imr_in = np.reshape(imr_in,(:,1))
        img_in = x[:][:][1]
        img_in = np.reshape(img_in,(:,1))
        imb_in = x[:][:][2]
        imb_in = np.reshape(imb_in,(:,1))
        
        predictions = self.neural_model(self.imr_input,
                                        self.img_input,
                                        self.imb_input,
                                        self.depth_input,
                                        self.gnd_input,
                                        self.obj_input,
                                        self.bld_input,
                                        self.veg_input,
                                        self.sky_input)

        load_weights = tf.train.Saver()

        if run==False:
            raise ValueError

        dir = 'models/full/' + run

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            load_weights.restore(sess,dir+'/fullmodel.ckpt')

            # taking only rgb as input
            depth_in = [[0]*self.size_input]
            gnd_in = [[0]*self.size_input]
            obj_in = [[0]*self.size_input]
            bld_in = [[0]*self.size_input]
            veg_in = [[0]*self.size_input]
            sky_in = [[0]*self.size_input]
            
            feed_dict = {self.imr_input:[imr_in],
                         self.img_input:[img_in],
                         self.imb_input:[imb_in],
                         self.depth_input:depth_in,
                         self.gnd_input:gnd_in,
                         self.obj_input:obj_in,
                         self.bld_input:bld_in,
                         self.veg_input:veg_in,
                         self.sky_input:sky_in}
            
            pred = sess.run(predictions,feed_dict=feed_dict)
            
            return np.asarray(pred)
            '''

    def evaluate_sequence(self,sequence,option=None,n_rnn_steps=None,frequency=None,run=None):

        if run == None:
            raise ValueError('no run ID given')

        if option == None:
            raise ValueError('no distortion option given')

        if frequency == None:
            raise ValueError('no distortion frequency given')

        if n_rnn_steps == None:
            raise ValueError('no number of rnn steps given')


        n_steps = len(sequence[0])

        label_data = sequence
        input_data = distort_test_sequences(deepcopy(label_data),n_rnn_steps=n_rnn_steps,option=option,frequency=frequency)

        input  = [self.imr_input,self.img_input,self.imb_input,self.depth_input,
                  self.gnd_input,self.obj_input,self.bld_input,self.veg_input,self.sky_input]
        output = self.network(input)

        # preparation of load model
        load_weights = tf.train.Saver()

        dir = 'models/full/' + run

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            load_weights.restore(sess,dir+'/fullmodel.ckpt') # runs from 06112017 it ist fullmodel_rnn

            error_rms = []
            error_rel = []

            for i in range(0,n_steps):

                depth_label = label_data[3][i]
                depth_label = depth_label[-1]

                imr_in,img_in,imb_in,depth_in,gnd_in,obj_in,bld_in,veg_in,sky_in = input_distortion(copy(input_data[0][i][-1]),
                                                                                                    copy(input_data[1][i][-1]),
                                                                                                    copy(input_data[2][i][-1]),
                                                                                                    copy(input_data[3][i][-1]),
                                                                                                    copy(input_data[4][i][-1]),
                                                                                                    copy(input_data[5][i][-1]),
                                                                                                    copy(input_data[6][i][-1]),
                                                                                                    copy(input_data[7][i][-1]),
                                                                                                    copy(input_data[8][i][-1]),
                                                                                                    resolution=(18,60),
                                                                                                    rnn=False,
                                                                                                    singleframe=True)

                feed_dict = {self.imr_input:imr_in,
                             self.img_input:img_in,
                             self.imb_input:imb_in,
                             self.depth_input:depth_in,
                             self.gnd_input:gnd_in,
                             self.obj_input:obj_in,
                             self.bld_input:bld_in,
                             self.veg_input:veg_in,
                             self.sky_input:sky_in}

                pred = sess.run(output,feed_dict=feed_dict)
                depth_pred = pred[3]



                inv_depth_pred = np.asarray(copy(depth_pred))
                inv_depth_label = np.asarray(copy(depth_label))



                gt = BR.invert_depth(inv_depth_label)
                est = BR.invert_depth(inv_depth_pred)

                error_rms.append(eval.rms_error(est,gt))
                error_rel.append(eval.relative_error(est,gt))

        sess.close()
        tf.reset_default_graph()

        return error_rms,error_rel

# running model




















