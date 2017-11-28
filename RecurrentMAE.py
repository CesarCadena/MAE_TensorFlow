# model based on 'Multi-modal Auto-Encoders as Joint Estimators for Robotics Scene Understanding' by Cadena et al.
# code developed by Silvan Weder
from collections import deque
import resource
from resource import getrusage


import tensorflow as tf
import numpy as np
import os
import evaluation_functions as eval
import basic_routines as BR

from load_data import load_training_data, load_validation_data, load_test_data
from input_distortion import input_distortion
from datetime import datetime
from copy import copy



# deleted blue_dc_layer_bias from loader (wrong name in full model)
# need to change that afterwards, when corrected full model is trained



class RecurrentMAE:

    def __init__(self,data_train,data_validate,resolution=(18,60)):

        self.data_train = data_train
        self.data_validate = data_validate

        self.height = resolution[0]
        self.width = resolution[1]

        self.size_input = self.height*self.width
        self.size_coding = 1024

        self.n_training_validations = 50

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

         # recurrent options

        self.n_rnn_steps = 5
        self.state_size = 1024


        # prepare data
        self.data_augmentation = True
        self.n_augmentations = 3

        self.prepare_training_data()
        self.prepare_validation_data()

        # flags
        self.flag_is_running = False

        self.placeholder_definition()

        # training options
        self.batch_size = 60
        self.n_batches = int(len(self.train_sequences)/self.batch_size)


        self.learning_rate = 1e-6
        self.hm_epochs = 1000


        # variables for overfitting detection
        self.of_det_gnd = deque([])
        self.of_det_obj = deque([])
        self.of_det_bld = deque([])
        self.of_det_veg = deque([])
        self.of_det_sky = deque([])


        self.gnd_dw = False
        self.obj_dw = False
        self.bld_dw = False
        self.veg_dw = False
        self.sky_dw = False

        # model savings
        self.saving = True
        now = datetime.now()

        self.folder_model = 'models/'
        self.folder_logs = 'logs/'

        self.mode = 'rnn/'
        self.run = now.strftime('%Y%m%d-%H%M%S')

        self.project_dir = './'
        self.model_dir = self.project_dir + self.folder_model + self.mode + self.run
        self.logs_dir = self.project_dir + self.folder_logs + self.mode + self.run
        self.load_dir = self.project_dir + self.folder_model + 'full/FullMAE/'

        os.mkdir(self.model_dir)
        os.mkdir(self.logs_dir)

        tf.app.flags.DEFINE_string('logs_dir',self.logs_dir,'where to store the logs')
        tf.app.flags.DEFINE_string('model_dir',self.model_dir,'where to store the trained model')

        self.FLAGS = tf.app.flags.FLAGS

    def placeholder_definition(self):

        # placeholder definition
        self.imr_input = tf.placeholder('float',[None,self.n_rnn_steps,self.size_input])
        self.img_input = tf.placeholder('float',[None,self.n_rnn_steps,self.size_input])
        self.imb_input = tf.placeholder('float',[None,self.n_rnn_steps,self.size_input])
        self.depth_input = tf.placeholder('float',[None,self.n_rnn_steps,self.size_input])
        self.gnd_input = tf.placeholder('float',[None,self.n_rnn_steps,self.size_input])
        self.obj_input = tf.placeholder('float',[None,self.n_rnn_steps,self.size_input])
        self.bld_input = tf.placeholder('float',[None,self.n_rnn_steps,self.size_input])
        self.veg_input = tf.placeholder('float',[None,self.n_rnn_steps,self.size_input])
        self.sky_input = tf.placeholder('float',[None,self.n_rnn_steps,self.size_input])

        self.depth_mask = tf.placeholder('float',[None,self.n_rnn_steps,self.size_input])

        self.imr_label = tf.placeholder('float',[None,self.n_rnn_steps,self.size_input])
        self.img_label = tf.placeholder('float',[None,self.n_rnn_steps,self.size_input])
        self.imb_label = tf.placeholder('float',[None,self.n_rnn_steps,self.size_input])
        self.depth_label = tf.placeholder('float',[None,self.n_rnn_steps,self.size_input])
        self.gnd_label = tf.placeholder('float',[None,self.n_rnn_steps,self.size_input])
        self.obj_label = tf.placeholder('float',[None,self.n_rnn_steps,self.size_input])
        self.bld_label = tf.placeholder('float',[None,self.n_rnn_steps,self.size_input])
        self.veg_label = tf.placeholder('float',[None,self.n_rnn_steps,self.size_input])
        self.sky_label = tf.placeholder('float',[None,self.n_rnn_steps,self.size_input])

        # rnn initial states
        self.init_states = tf.placeholder('float',[None,self.state_size])

    def augment_data(self,imr,img,imb,depth,gnd,obj,bld,veg,sky):

        imr_aug = []
        img_aug = []
        imb_aug = []
        depth_aug = []
        gnd_aug = []
        obj_aug = []
        bld_aug = []
        veg_aug = []
        sky_aug = []

        for series in range(0,self.batch_size):

            imr_series_aug = copy(imr[series])
            img_series_aug = copy(img[series])
            imb_series_aug = copy(imb[series])
            depth_series_aug = copy(depth[series])
            gnd_series_aug = copy(gnd[series])
            obj_series_aug = copy(obj[series])
            bld_series_aug = copy(bld[series])
            veg_series_aug = copy(veg[series])
            sky_series_aug = copy(sky[series])

            u = np.random.randint(0,8)

            for rnn_step in range(0,self.n_rnn_steps):


                zeros = np.zeros(np.asarray(imr_series_aug[rnn_step]).shape)

                if u == 1:

                    # only rgb
                    if rnn_step < 2:

                        depth_series_aug[rnn_step] = zeros
                        gnd_series_aug[rnn_step] = zeros
                        obj_series_aug[rnn_step] = zeros
                        bld_series_aug[rnn_step] = zeros
                        veg_series_aug[rnn_step] = zeros
                        sky_series_aug[rnn_step] = zeros

                    if rnn_step < 4:

                        depth_series_aug[rnn_step] = zeros

                if u == 2:

                    if rnn_step < 2:

                        depth_series_aug[rnn_step] = zeros
                        gnd_series_aug[rnn_step] = zeros
                        obj_series_aug[rnn_step] = zeros
                        bld_series_aug[rnn_step] = zeros
                        veg_series_aug[rnn_step] = zeros
                        sky_series_aug[rnn_step] = zeros

                    if rnn_step < 4:

                        gnd_series_aug[rnn_step] = zeros
                        obj_series_aug[rnn_step] = zeros
                        bld_series_aug[rnn_step] = zeros
                        veg_series_aug[rnn_step] = zeros
                        sky_series_aug[rnn_step] = zeros


                if u == 3:

                    if rnn_step%2 == 0:

                        depth_series_aug[rnn_step] = zeros
                        gnd_series_aug[rnn_step] = zeros
                        obj_series_aug[rnn_step] = zeros
                        bld_series_aug[rnn_step] = zeros
                        veg_series_aug[rnn_step] = zeros
                        sky_series_aug[rnn_step] = zeros

                    else:
                        depth_series_aug[rnn_step] = zeros


                if u == 5:

                    if rnn_step%2 == 0:

                        depth_series_aug[rnn_step] = zeros
                        gnd_series_aug[rnn_step] = zeros
                        obj_series_aug[rnn_step] = zeros
                        bld_series_aug[rnn_step] = zeros
                        veg_series_aug[rnn_step] = zeros
                        sky_series_aug[rnn_step] = zeros

                    else:

                        gnd_series_aug[rnn_step] = zeros
                        obj_series_aug[rnn_step] = zeros
                        bld_series_aug[rnn_step] = zeros
                        veg_series_aug[rnn_step] = zeros
                        sky_series_aug[rnn_step] = zeros

                if u == 6:

                    depth_series_aug[rnn_step] = zeros
                    gnd_series_aug[rnn_step] = zeros
                    obj_series_aug[rnn_step] = zeros
                    bld_series_aug[rnn_step] = zeros
                    veg_series_aug[rnn_step] = zeros
                    sky_series_aug[rnn_step] = zeros

                if u == 7:

                    depth_series_aug[rnn_step] = zeros

                if u == 8:
                    pass


            imr_aug.append(copy(imr_series_aug))
            imb_aug.append(copy(img_series_aug))
            img_aug.append(copy(imb_series_aug))
            depth_aug.append(copy(depth_series_aug))
            gnd_aug.append(copy(gnd_series_aug))
            obj_aug.append(copy(obj_series_aug))
            bld_aug.append(copy(bld_series_aug))
            veg_aug.append(copy(veg_series_aug))
            sky_aug.append(copy(sky_series_aug))

        return np.asarray(imr_aug),np.asarray(img_aug),np.asarray(imb_aug),\
               np.asarray(depth_aug),np.asarray(gnd_aug),np.asarray(obj_aug),\
               np.asarray(bld_aug),np.asarray(veg_aug),np.asarray(sky_aug)

    def prepare_training_data(self):
        '''
        Function for bringing the training data into a form that suits the training process
        :return:
        '''

        self.train_sequences = []

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

        self.imr_train_label = []
        self.img_train_label = []
        self.imb_train_label = []
        self.depth_train_label = []
        self.gnd_train_label = []
        self.obj_train_label = []
        self.bld_train_label = []
        self.veg_train_label = []
        self.sky_train_label = []



        for sequence in range(0,len(self.data_train)):
            for frame in range(0,len(self.data_train[sequence])):

                frame_series = []

                for timestep in range(0,self.n_rnn_steps):

                    offset = self.n_rnn_steps-1-timestep
                    index = frame - offset

                    if index < 0:

                        zero_padding = (-1,-1)
                        frame_series.append(zero_padding)

                    else:

                        indices = (sequence,index)
                        frame_series.append(indices)

                self.train_sequences.append(copy(frame_series))


        # randomly shuffle input sequences
        rand_indices = np.arange(len(self.train_sequences)).astype(int)
        np.random.shuffle(rand_indices)

        self.train_sequences = np.asarray(self.train_sequences)[rand_indices]

        '''
        for i in range(0,len(self.data_train)):
            for j in range(0,len(self.data_train[i])):

                imr_series = []
                img_series = []
                imb_series = []
                depth_series = []
                depth_mask_series = []
                gnd_series = []
                obj_series = []
                bld_series = []
                veg_series = []
                sky_series = []

                for k in range(0,self.n_rnn_steps):
                    offset = self.n_rnn_steps-1-k
                    check = j - offset
                    if check < 0:

                        zero_padding = np.zeros((self.size_input,))

                        imr_series.append(zero_padding)
                        img_series.append(zero_padding)
                        imb_series.append(zero_padding)
                        depth_series.append(zero_padding)
                        depth_mask_series.append(zero_padding)
                        gnd_series.append(zero_padding)
                        obj_series.append(zero_padding)
                        bld_series.append(zero_padding)
                        veg_series.append(zero_padding)
                        sky_series.append(zero_padding)

                    else:

                        imr_series.append(i[j-offset]['xcr1']/255.)
                        img_series.append(i[j-offset]['xcg1']/255.)
                        imb_series.append(i[j-offset]['xcb1']/255.)
                        depth_series.append(i[j-offset]['xid1'])
                        depth_mask_series.append(i[j-offset]['xmask1'])
                        gnd_series.append((i[j-offset]['sem1']==1).astype(int))
                        obj_series.append((i[j-offset]['sem1']==2).astype(int))
                        bld_series.append((i[j-offset]['sem1']==3).astype(int))
                        veg_series.append((i[j-offset]['sem1']==4).astype(int))
                        sky_series.append((i[j-offset]['sem1']==5).astype(int))


                self.imr_train.append(copy(imr_series))
                self.img_train.append(copy(img_series))
                self.imb_train.append(copy(imb_series))
                self.depth_train.append(copy(depth_series))
                self.depth_mask_train.append(copy(depth_mask_series))
                self.gnd_train.append(copy(gnd_series))
                self.obj_train.append(copy(obj_series))
                self.bld_train.append(copy(bld_series))
                self.veg_train.append(copy(veg_series))
                self.sky_train.append(copy(sky_series))

                self.imr_train_label.append(copy(imr_series))
                self.img_train_label.append(copy(img_series))
                self.imb_train_label.append(copy(imb_series))
                self.depth_train_label.append(copy(depth_series))
                self.gnd_train_label.append(copy(gnd_series))
                self.obj_train_label.append(copy(obj_series))
                self.bld_train_label.append(copy(bld_series))
                self.veg_train_label.append(copy(veg_series))
                self.sky_train_label.append(copy(sky_series))


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

        self.imr_train_label = np.asarray(self.imr_train_label)[rand_indices]
        self.img_train_label = np.asarray(self.img_train_label)[rand_indices]
        self.imb_train_label = np.asarray(self.imb_train_label)[rand_indices]
        self.depth_train_label = np.asarray(self.depth_train_label)[rand_indices]
        self.gnd_train_label = np.asarray(self.gnd_train_label)[rand_indices]
        self.obj_train_label = np.asarray(self.obj_train_label)[rand_indices]
        self.bld_train_label = np.asarray(self.bld_train_label)[rand_indices]
        self.veg_train_label = np.asarray(self.veg_train_label)[rand_indices]
        self.sky_train_label = np.asarray(self.sky_train_label)[rand_indices]

        self.depth_mask_train = np.asarray(self.depth_mask_train)[rand_indices]
        
        '''

    def prepare_validation_data(self):

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


        for i in self.data_validate:
            for j in range(len(i)):

                imr_series = []
                img_series = []
                imb_series = []
                depth_series = []
                depth_mask_series = []
                gnd_series = []
                obj_series = []
                bld_series = []
                veg_series = []
                sky_series = []

                for k in range(0,self.n_rnn_steps):

                    offset = self.n_rnn_steps-1-k
                    check = j - offset

                    if check < 0:

                        zero_padding = np.zeros((self.size_input,))

                        imr_series.append(zero_padding)
                        img_series.append(zero_padding)
                        imb_series.append(zero_padding)
                        depth_series.append(zero_padding)
                        depth_mask_series.append(zero_padding)
                        gnd_series.append(zero_padding)
                        obj_series.append(zero_padding)
                        bld_series.append(zero_padding)
                        veg_series.append(zero_padding)
                        sky_series.append(zero_padding)

                    else:

                        imr_series.append(i[j-offset]['xcr1']/255.)
                        img_series.append(i[j-offset]['xcg1']/255.)
                        imb_series.append(i[j-offset]['xcb1']/255.)
                        depth_series.append(i[j-offset]['xid1'])
                        depth_mask_series.append(i[j-offset]['xmask1'])
                        gnd_series.append((i[j-offset]['sem1']==1).astype(int))
                        obj_series.append((i[j-offset]['sem1']==2).astype(int))
                        bld_series.append((i[j-offset]['sem1']==3).astype(int))
                        veg_series.append((i[j-offset]['sem1']==4).astype(int))
                        sky_series.append((i[j-offset]['sem1']==5).astype(int))


                self.imr_val.append(copy(imr_series))
                self.img_val.append(copy(img_series))
                self.imb_val.append(copy(imb_series))
                self.depth_val.append(copy(depth_series))
                self.depth_mask_val.append(copy(depth_mask_series))
                self.gnd_val.append(copy(gnd_series))
                self.obj_val.append(copy(obj_series))
                self.bld_val.append(copy(bld_series))
                self.veg_val.append(copy(veg_series))
                self.sky_val.append(copy(sky_series))

    def prepare_test_data(self):

        # prepare test data containers
        self.imr_test = []
        self.img_test = []
        self.imb_test = []
        self.depth_test = []
        self.depth_mask_test = []
        self.gnd_test = []
        self.obj_test = []
        self.bld_test = []
        self.veg_test = []
        self.sky_test = []


        for i in self.data_test:
            for j in range(len(i)):
                imr_series = []
                img_series = []
                imb_series = []
                depth_series = []
                depth_mask_series = []
                gnd_series = []
                obj_series = []
                bld_series = []
                veg_series = []
                sky_series = []

                for k in range(0,self.n_rnn_steps):

                    offset = self.n_rnn_steps-1-k
                    check = j - offset

                    if check < 0:

                        zero_padding = np.zeros((self.size_input,))

                        imr_series.append(zero_padding)
                        img_series.append(zero_padding)
                        imb_series.append(zero_padding)
                        depth_series.append(zero_padding)
                        depth_mask_series.append(zero_padding)
                        gnd_series.append(zero_padding)
                        obj_series.append(zero_padding)
                        bld_series.append(zero_padding)
                        veg_series.append(zero_padding)
                        sky_series.append(zero_padding)

                    else:

                        imr_series.append(i[j-offset]['xcr1']/255.)
                        img_series.append(i[j-offset]['xcg1']/255.)
                        imb_series.append(i[j-offset]['xcb1']/255.)
                        depth_series.append(i[j-offset]['xid1'])
                        depth_mask_series.append(i[j-offset]['xmask1'])
                        gnd_series.append((i[j-offset]['sem1']==1).astype(int))
                        obj_series.append((i[j-offset]['sem1']==2).astype(int))
                        bld_series.append((i[j-offset]['sem1']==3).astype(int))
                        veg_series.append((i[j-offset]['sem1']==4).astype(int))
                        sky_series.append((i[j-offset]['sem1']==5).astype(int))

                self.imr_test.append(copy(imr_series))
                self.img_test.append(copy(img_series))
                self.imb_test.append(copy(imb_series))
                self.depth_test.append(copy(depth_series))
                self.gnd_test.append(copy(gnd_series))
                self.obj_test.append(copy(obj_series))
                self.bld_test.append(copy(bld_series))
                self.veg_test.append(copy(veg_series))
                self.sky_test.append(copy(sky_series))

        # randomly shuffle input frames
        rand_indices = np.arange(len(self.imr_test)).astype(int)
        np.random.shuffle(rand_indices)

        self.imr_test = np.asarray(self.imr_test)[rand_indices]
        self.img_test = np.asarray(self.img_test)[rand_indices]
        self.imb_test = np.asarray(self.imb_test)[rand_indices]
        self.depth_test = np.asarray(self.depth_test)[rand_indices]
        self.gnd_test = np.asarray(self.gnd_test)[rand_indices]
        self.obj_test = np.asarray(self.obj_test)[rand_indices]
        self.bld_test = np.asarray(self.bld_test)[rand_indices]
        self.veg_test = np.asarray(self.veg_test)[rand_indices]
        self.sky_test = np.asarray(self.sky_test)[rand_indices]

    def encoding_network(self, imr, img, imb, depth, gnd, obj, bld, veg, sky):

        '''
        :param imr: red channel of rgb image
        :param img: green channel of rgb image
        :param imb: blue channel of rgb image
        :param depth: inv depth values for image (loss is only computed at accepted points)
        :param gnd: binary image for class ground in image semantics
        :param obj: binary image for class object in image semantics
        :param bld: binary image for class building in image semantics
        :param veg: binary image for class vegetation in image semantics
        :param sky: binary image for class sky in image semantics
        :return: reconstructed images for all input modalities
        '''

        # definition of all variables in the neural network
        # list to store all layers of the MAE neural network
        self.layers = []

        with tf.variable_scope('Encoder') as encoder:
            # semantics weights
            self.gnd_ec_layer = {'weights':tf.Variable(tf.random_normal([self.size_input,self.size_coding],stddev=0.01),name='gnd_ec_layer_weights'),
                                 'bias':tf.Variable(tf.zeros([self.size_coding]),name='gnd_ec_layer_bias')}
            self.layers.append(self.gnd_ec_layer)

            self.obj_ec_layer = {'weights':tf.Variable(tf.random_normal([self.size_input,self.size_coding],stddev=0.01),name='obj_ec_layer_weights'),
                                 'bias':tf.Variable(tf.zeros([self.size_coding]),name='obj_ec_layer_bias')}
            self.layers.append(self.obj_ec_layer)

            self.bld_ec_layer = {'weights':tf.Variable(tf.random_normal([self.size_input,self.size_coding],stddev=0.01),name='bld_ec_layer_weights'),
                                 'bias':tf.Variable(tf.zeros([self.size_coding]),name='bld_ec_layer_bias')}
            self.layers.append(self.bld_ec_layer)

            self.veg_ec_layer = {'weights':tf.Variable(tf.random_normal([self.size_input,self.size_coding],stddev=0.01),name='veg_ec_layer_weights'),
                                 'bias':tf.Variable(tf.zeros([self.size_coding]),name='veg_ec_layer_bias')}
            self.layers.append(self.veg_ec_layer)

            self.sky_ec_layer = {'weights':tf.Variable(tf.random_normal([self.size_input,self.size_coding],stddev=0.01),name='sky_ec_layer_weights'),
                                 'bias':tf.Variable(tf.zeros([self.size_coding]),name='sky_ec_layer_bias')}
            self.layers.append(self.sky_ec_layer)

            # semantics encoding
            self.sem_ec_layer = {'weights':tf.Variable(tf.random_normal([5 * self.size_coding, self.size_coding], stddev=0.01), name='sem_ec_layer_weights'),
                                 'bias' : tf.Variable(tf.zeros([self.size_coding]),name='sem_ec_layer_bias')}
            self.layers.append(self.sem_ec_layer)

            # depth and rgb encoding weights
            self.red_ec_layer = {'weights':tf.Variable(tf.random_normal([self.size_input,self.size_coding],stddev=0.01),name='red_ec_layer_weights'),
                                 'bias':tf.Variable(tf.zeros([self.size_coding]),name='red_ec_layer_bias')}
            self.layers.append(self.red_ec_layer)

            self.green_ec_layer = {'weights':tf.Variable(tf.random_normal([self.size_input,self.size_coding],stddev=0.01),name='green_ec_layer_weights'),
                                   'bias':tf.Variable(tf.zeros([self.size_coding]),name='green_ec_layer_bias')}
            self.layers.append(self.green_ec_layer)

            self.blue_ec_layer = {'weights':tf.Variable(tf.random_normal([self.size_input,self.size_coding],stddev=0.01),name='blue_ec_layer_weights'),
                                  'bias':tf.Variable(tf.zeros([self.size_coding]),name='blue_ec_layer_bias')}
            self.layers.append(self.blue_ec_layer)

            self.depth_ec_layer = {'weights':tf.Variable(tf.random_normal([self.size_input,self.size_coding],stddev=0.01),name='depth_ec_layer_weights'),
                                   'bias':tf.Variable(tf.zeros([self.size_coding]),name='depth_ec_layer_bias')}
            self.layers.append(self.depth_ec_layer)

             # full encoding

            self.full_ec_layer = {'weights':tf.Variable(tf.random_normal([5*self.size_coding,self.size_coding],stddev=0.01),name='full_ec_layer_weights'),
                                  'bias' : tf.Variable(tf.zeros([self.size_coding]),name='full_ec_layer_bias')}
            self.layers.append(self.full_ec_layer)

            self.encoder_variables = [v for v in tf.global_variables() if v.name.startswith(encoder.name)]

        # split the input placeholder according to the length of the sequence

        imr_series = tf.split(imr,self.n_rnn_steps,axis=1)
        img_series = tf.split(img,self.n_rnn_steps,axis=1)
        imb_series = tf.split(imb,self.n_rnn_steps,axis=1)
        depth_series = tf.split(depth,self.n_rnn_steps,axis=1)
        gnd_series = tf.split(gnd,self.n_rnn_steps,axis=1)
        obj_series = tf.split(obj,self.n_rnn_steps,axis=1)
        bld_series = tf.split(bld,self.n_rnn_steps,axis=1)
        veg_series = tf.split(veg,self.n_rnn_steps,axis=1)
        sky_series = tf.split(sky,self.n_rnn_steps,axis=1)

        # output container definition
        output = []

        for i in range(0,self.n_rnn_steps):

            imr_in = tf.squeeze(imr_series[i],axis=1)
            img_in = tf.squeeze(img_series[i],axis=1)
            imb_in = tf.squeeze(imb_series[i],axis=1)
            depth_in = tf.squeeze(depth_series[i],axis=1)
            gnd_in = tf.squeeze(gnd_series[i],axis=1)
            obj_in = tf.squeeze(obj_series[i],axis=1)
            bld_in = tf.squeeze(bld_series[i],axis=1)
            veg_in = tf.squeeze(veg_series[i],axis=1)
            sky_in = tf.squeeze(sky_series[i],axis=1)

            # semantics neurons (relu activation)
            self.gnd_encoding = tf.add(tf.matmul(gnd_in,self.gnd_ec_layer['weights']),
                                       self.gnd_ec_layer['bias'])
            self.gnd_encoding = tf.nn.relu(self.gnd_encoding)

            self.obj_encoding = tf.add(tf.matmul(obj_in,self.obj_ec_layer['weights']),
                                       self.obj_ec_layer['bias'])
            self.obj_encoding = tf.nn.relu(self.obj_encoding)

            self.bld_encoding = tf.add(tf.matmul(bld_in,self.bld_ec_layer['weights']),
                                       self.bld_ec_layer['bias'])
            self.bld_encoding = tf.nn.relu(self.bld_encoding)

            self.veg_encoding = tf.add(tf.matmul(veg_in,self.veg_ec_layer['weights']),
                                       self.veg_ec_layer['bias'])
            self.veg_encoding = tf.nn.relu(self.veg_encoding)

            self.sky_encoding = tf.add(tf.matmul(sky_in,self.sky_ec_layer['weights']),
                                       self.sky_ec_layer['bias'])
            self.sky_encoding = tf.nn.relu(self.sky_encoding)

            # semantics concatenate
            self.sem_concat = tf.concat([self.gnd_encoding,
                                         self.obj_encoding,
                                         self.bld_encoding,
                                         self.veg_encoding,
                                         self.sky_encoding],
                                        axis=1)



            # semantics neuron (relu activation)
            self.sem_encoding = tf.add(tf.matmul(self.sem_concat, self.sem_ec_layer['weights']),
                                       self.sem_ec_layer['bias'])
            self.sem_encoding = tf.nn.relu(self.sem_encoding)




            # depth and rgb neurons (relu activation)
            self.red_encoding = tf.add(tf.matmul(imr_in,self.red_ec_layer['weights']),
                                       self.red_ec_layer['bias'])
            self.red_encoding = tf.nn.relu(self.red_encoding)

            self.green_encoding = tf.add(tf.matmul(img_in,self.green_ec_layer['weights']),
                                         self.green_ec_layer['bias'])
            self.green_encoding = tf.nn.relu(self.green_encoding)

            self.blue_encoding = tf.add(tf.matmul(imb_in,self.blue_ec_layer['weights']),
                                        self.blue_ec_layer['bias'])
            self.blue_encoding = tf.nn.relu(self.blue_encoding)

            self.depth_encoding = tf.add(tf.matmul(depth_in,self.depth_ec_layer['weights']),
                                         self.depth_ec_layer['bias'])

            # full concatenation

            self.full_concat = tf.concat([self.depth_encoding,self.red_encoding,self.green_encoding,self.blue_encoding,self.sem_encoding],
                                         axis=1)

            # full encoding neurons

            self.full_encoding = tf.add(tf.matmul(self.full_concat,self.full_ec_layer['weights']),
                                        self.full_ec_layer['bias'])
            self.full_encoding = tf.nn.relu(self.full_encoding)

            output.append(self.full_encoding)

        return output

    def decoding_network(self,inputs):

        # definition of variables for whole decoding network
        # full decoding

        with tf.variable_scope('Decoder') as decoder:

            self.full_dc_layer = {'weights':tf.Variable(tf.random_normal([self.size_coding,5*self.size_coding],stddev=0.01),name='full_dc_layer_weights'),
                                  'bias':tf.Variable(tf.zeros([5*self.size_coding]),name='full_dc_layer_bias')}
            self.layers.append(self.full_dc_layer)

            # rgb and depth decoding layers

            self.red_dc_layer = {'weights':tf.Variable(tf.random_normal([self.size_coding,self.size_input],stddev=0.01),name='red_dc_layer_weights'),
                                 'bias':tf.Variable(tf.zeros([self.size_input]),name='red_dc_layer_bias')}
            self.layers.append(self.red_dc_layer)

            self.green_dc_layer = {'weights':tf.Variable(tf.random_normal([self.size_coding,self.size_input],stddev=0.01),name='green_dc_layer_weights'),
                                   'bias':tf.Variable(tf.zeros([self.size_input]),name='green_dc_layer_bias')}
            self.layers.append(self.green_dc_layer)

            self.blue_dc_layer = {'weights':tf.Variable(tf.random_normal([self.size_coding,self.size_input],stddev=0.01),name='blue_dc_layer_weights'),
                                  'bias':tf.Variable(tf.zeros([self.size_input]),name='blue_dc_layer_bias')}
            self.layers.append(self.blue_dc_layer)

            self.depth_dc_layer = {'weights':tf.Variable(tf.random_normal([self.size_coding,self.size_input],stddev=0.01),name='depth_dc_layer_weights'),
                                   'bias':tf.Variable(tf.zeros([self.size_input]),name='depth_dc_layer_bias')}
            self.layers.append(self.depth_dc_layer)

            # decoding layer full semantics

            self.sem_dc_layer = {'weights':tf.Variable(tf.random_normal([self.size_coding, 5 * self.size_coding], stddev=0.01), name='sem_dc_layer_weights'),
                                      'bias':tf.Variable(tf.zeros([5*self.size_coding]),name='full_sem_dc_layer_bias')}
            self.layers.append(self.sem_dc_layer)

            # decoding layers semantics

            self.gnd_dc_layer = {'weights':tf.Variable(tf.random_normal([self.size_coding,self.size_input],stddev=0.01),name='gnd_dc_layer_weights'),
                                 'bias':tf.Variable(tf.zeros([self.size_input]),name='gnd_dc_layer_bias')}
            self.layers.append(self.gnd_dc_layer)

            self.obj_dc_layer = {'weights':tf.Variable(tf.random_normal([self.size_coding,self.size_input],stddev=0.01),name='obj_dc_layer_weights'),
                                 'bias':tf.Variable(tf.zeros([self.size_input]),name='obj_dc_layer_bias')}
            self.layers.append(self.obj_dc_layer)

            self.bld_dc_layer = {'weights':tf.Variable(tf.random_normal([self.size_coding,self.size_input],stddev=0.01),name='bld_dc_layer_weights'),
                                 'bias':tf.Variable(tf.zeros([self.size_input]),name='bld_dc_layer_bias')}
            self.layers.append(self.bld_dc_layer)

            self.veg_dc_layer = {'weights':tf.Variable(tf.random_normal([self.size_coding,self.size_input],stddev=0.01),name='veg_dc_layer_weights'),
                                 'bias':tf.Variable(tf.zeros([self.size_input]),name='veg_dc_layer_bias')}
            self.layers.append(self.veg_dc_layer)

            self.sky_dc_layer = {'weights':tf.Variable(tf.random_normal([self.size_coding,self.size_input],stddev=0.01),name='sky_dc_layer_weights'),
                                 'bias':tf.Variable(tf.zeros([self.size_input]),name='sky_dc_layer_bias')}
            self.layers.append(self.sky_dc_layer)

            self.decoder_variables = [v for v in tf.global_variables() if v.name.startswith(decoder.name)]

        # full decoding layer

        self.full_decoding = tf.add(tf.matmul(inputs,self.full_dc_layer['weights']),self.full_dc_layer['bias'])
        self.full_decoding = tf.nn.relu(self.full_decoding)

        # slicing full decoding

        self.depth_full_dc,self.red_full_dc,self.green_full_dc,self.blue_full_dc,self.sem_full_dc = tf.split(self.full_decoding,5,1)


        # decoding neurons

        self.red_output = tf.add(tf.matmul(self.red_full_dc,self.red_dc_layer['weights']),
                                           self.red_dc_layer['bias'])
        self.red_output = tf.sigmoid(self.red_output)

        self.green_output = tf.add(tf.matmul(self.green_full_dc, self.green_dc_layer['weights']),
                                           self.green_dc_layer['bias'])
        self.green_output = tf.sigmoid(self.green_output)

        self.blue_output = tf.add(tf.matmul(self.blue_full_dc, self.blue_dc_layer['weights']),
                                          self.blue_dc_layer['bias'])
        self.blue_output = tf.sigmoid(self.blue_output)

        self.depth_output = tf.add(tf.matmul(self.depth_full_dc, self.depth_dc_layer['weights']),
                                           self.depth_dc_layer['bias'])
        self.depth_output = tf.nn.relu(self.depth_output)

        imr_out = self.red_output
        img_out = self.green_output
        imb_out = self.blue_output
        depth_out = self.depth_output

        # decoding neurons full semantics

        self.full_sem = tf.add(tf.matmul(self.sem_full_dc, self.sem_dc_layer['weights']),
                               self.sem_dc_layer['bias'])
        self.full_sem = tf.nn.relu(self.full_sem)

        # splitting full semantics

        self.gnd_dc, self.obj_dc, self.bld_dc, self.veg_dc, self.sky_dc = tf.split(self.full_sem,5,axis=1)

        # decoding neurons semantics

        self.gnd_output = tf.add(tf.matmul(self.gnd_dc,self.gnd_dc_layer['weights']),
                                         self.gnd_dc_layer['bias'])
        self.gnd_output = tf.sigmoid(self.gnd_output)

        self.obj_output = tf.add(tf.matmul(self.obj_dc,self.obj_dc_layer['weights']),
                                         self.obj_dc_layer['bias'])
        self.obj_output = tf.sigmoid(self.obj_output)

        self.bld_output = tf.add(tf.matmul(self.bld_dc,self.bld_dc_layer['weights']),
                                         self.bld_dc_layer['bias'])
        self.bld_output = tf.sigmoid(self.bld_output)

        self.veg_output = tf.add(tf.matmul(self.veg_dc,self.veg_dc_layer['weights']),
                                         self.veg_dc_layer['bias'])
        self.veg_output = tf.sigmoid(self.veg_output)

        self.sky_output = tf.add(tf.matmul(self.sky_dc,self.sky_dc_layer['weights']),
                                         self.sky_dc_layer['bias'])
        self.sky_output = tf.sigmoid(self.sky_output)

        gnd_out = self.gnd_output
        obj_out = self.obj_output
        bld_out = self.bld_output
        veg_out = self.veg_output
        sky_out = self.sky_output

        output = [imr_out,img_out,imb_out,depth_out,gnd_out,obj_out,bld_out,veg_out,sky_out]
        return output

    def Basic_RNN(self, inputs):

        # container for recurrent weights
        self.rnn_weights_H = []
        self.rnn_weights_W = []
        self.rnn_bias_B = []

        # set state size
        state_size = self.state_size

        with tf.variable_scope('RNN') as rnn:
            for i in range(0,self.n_rnn_steps):

                # initialization of recurrent weights
                self.rnn_weights_H.append(tf.Variable(tf.zeros([state_size,state_size],dtype=tf.float32),name='rnn_H_' + str(i)))
                self.rnn_weights_W.append(tf.Variable(tf.concat([tf.diag(tf.ones([self.size_coding])),
                                                                 tf.zeros([self.size_coding,self.state_size-self.size_coding])],axis=1),
                                                      dtype=tf.float32,
                                                      name='rnn_W_' + str(i)))

                self.rnn_bias_B.append(tf.Variable(tf.zeros([state_size],dtype=tf.float32),name='rnn_B_' + str(i)))

            # initialization of weights from current timestep
            self.rnn_weights_W.append(tf.Variable(tf.concat([tf.diag(tf.ones([self.size_coding])),
                                                             tf.zeros([self.size_coding,self.state_size-self.size_coding])],axis=1),
                                                  dtype=tf.float32,
                                                  name='rnn_W_out'))

            self.rnn_bias_B.append(tf.Variable(tf.zeros([state_size],dtype=tf.float32),name='rnn_B_' + str(i)))

            self.rnn_weights_V = tf.Variable(tf.concat([tf.diag(tf.ones([self.size_coding])),
                                                        tf.zeros([self.state_size-self.size_coding,self.size_coding])],axis=0),
                                             dtype=tf.float32,
                                             name='rnn_V_out')

            # get all variables of rnn network
            self.rnn_variables = [v for v in tf.global_variables() if v.name.startswith(rnn.name)]

        # initialize state of recurrent network from initializing placeholder
        state = self.init_states

        # running recurrent layer
        for i in range(0,self.n_rnn_steps):
            state = tf.matmul(tf.add(tf.add(state,tf.matmul(inputs[i],self.rnn_weights_W[i])),self.rnn_bias_B[i]),self.rnn_weights_H[i])

        output = tf.matmul(tf.add(tf.add(tf.matmul(inputs[-1],self.rnn_weights_W[-1]),state),self.rnn_bias_B[-1]),self.rnn_weights_V)

        return output

    def LSTM_RNN(self,inputs):

        # options
        size_states = 1024
        size_coding = 1024

        # define initializer for input weights
        if size_states == size_coding:
            initializer_U = 0.0001*tf.diag(tf.ones([size_coding]))
        else:
            initializer_U = tf.concat([tf.diag(tf.ones([size_coding])),tf.zeros([size_coding,size_states-size_coding])],axis=1)

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
        with tf.variable_scope('LSTM') as lstm:

            for step in range(0,self.n_rnn_steps):

                # forget gate
                self.b_f.append(tf.get_variable(name='b_f_'+str(step),
                                                shape=[size_states],
                                                dtype=tf.float32,
                                                initializer=tf.zeros_initializer()))

                self.U_f.append(tf.get_variable(name='U_f_'+str(step),
                                                dtype=tf.float32,
                                                initializer=initializer_U))

                self.W_f.append(tf.get_variable(name='W_f_'+str(step),
                                                shape=[size_states,size_states],
                                                dtype=tf.float32,
                                                initializer=tf.zeros_initializer()))


                # internal state
                self.b.append(tf.get_variable(name='b_'+str(step),
                                              shape=[size_states],
                                              dtype=tf.float32,
                                              initializer=tf.zeros_initializer()))

                self.U.append(tf.get_variable(name='U_'+str(step),
                                              dtype=tf.float32,
                                              initializer=initializer_U))

                self.W.append(tf.get_variable(name='W_'+str(step),
                                              shape=[size_states,size_states],
                                              dtype=tf.float32,
                                              initializer=tf.zeros_initializer()))

                # external input gate
                self.b_g.append(tf.get_variable(name='b_g_'+str(step),
                                                shape=[size_states],
                                                dtype=tf.float32,
                                                initializer=tf.zeros_initializer()))

                self.U_g.append(tf.get_variable(name='U_g_'+str(step),
                                                dtype=tf.float32,
                                                initializer=initializer_U))

                self.W_g.append(tf.get_variable(name='W_g_'+str(step),
                                                shape=[size_states,size_states],
                                                dtype=tf.float32,
                                                initializer=tf.zeros_initializer()))

                # output gate
                self.b_o.append(tf.get_variable(name='b_o_'+str(step),
                                                shape=[size_states],
                                                dtype=tf.float32,
                                                initializer=tf.zeros_initializer()))

                self.U_o.append(tf.get_variable(name='U_o_'+str(step),
                                                dtype=tf.float32,
                                                initializer=initializer_U))

                self.W_o.append(tf.get_variable(name='W_o_'+str(step),
                                                shape=[size_states,size_states],
                                                dtype=tf.float32,
                                                initializer=tf.zeros_initializer()))

            self.rnn_variables = [v for v in tf.global_variables() if v.name.startswith(lstm.name)]

        # state-to-coding weights initializer
        if size_states == size_coding:
            initializer_O = tf.diag(tf.ones([size_coding]))
        else:
            initializer_O = tf.concat([tf.diag(tf.ones([size_coding])),tf.zeros([size_states-size_coding,size_coding])],axis=0)

        # state-to-coding weights
        self.O_w = tf.get_variable(name='O_w',
                                   dtype=tf.float32,
                                   initializer=initializer_O)

        self.O_b = tf.get_variable(name='O_b',
                                   shape=[size_coding],
                                   dtype=tf.float32,
                                   initializer=tf.zeros_initializer())

        # state initialization
        h_t = self.init_states
        s_t = self.init_states

        # cell definition
        for step in range(0,self.n_rnn_steps):

            # forget gate
            f_t = tf.sigmoid(self.b_f[step] + tf.matmul(inputs[step],self.U_f[step]) + tf.matmul(h_t,self.W_f[step]))

            # internal states
            i_t = tf.sigmoid(self.b_g[step] + tf.matmul(inputs[step],self.U_g[step]) + tf.matmul(h_t,self.W_g[step]))

            # external intput gate
            g_t = tf.tanh(self.b[step] + tf.matmul(inputs[step],self.U[step]) + tf.matmul(h_t,self.W[step]))

            # memory update
            s_t = tf.multiply(f_t,s_t) + tf.multiply(i_t,g_t)

            # output gate
            o_t = tf.sigmoid(self.b_o[step] + tf.matmul(inputs[step],self.U_o[step]) + tf.matmul(h_t,self.W_o[step]))

            # state update
            h_t = tf.multiply(o_t,tf.tanh(s_t))


        # reconstruct coding
        output = tf.add(tf.matmul(h_t,self.O_w),self.O_b)

        return output

    def network(self, input):


        encoding = self.encoding_network(input[0],
                                         input[1],
                                         input[2],
                                         input[3],
                                         input[4],
                                         input[5],
                                         input[6],
                                         input[7],
                                         input[8])

        outputs = self.Basic_RNN(encoding)

        output = self.decoding_network(outputs)

        return output

    def collect_variables(self):

        # collect all variables for weight regularization
        for i in self.layers:
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, i['weights'])
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, i['bias'])

    def overfitting_detection(self,val_losses,epoch):

        window_size = 10

        if len(self.of_det_gnd) < window_size:
            self.of_det_gnd.append(val_losses[0])
            self.of_det_obj.append(val_losses[1])
            self.of_det_bld.append(val_losses[2])
            self.of_det_veg.append(val_losses[3])
            self.of_det_sky.append(val_losses[4])

        else:
            gnd_avg = sum(self.of_det_gnd)/float(window_size)
            obj_avg = sum(self.of_det_obj)/float(window_size)
            bld_avg = sum(self.of_det_bld)/float(window_size)
            veg_avg = sum(self.of_det_veg)/float(window_size)
            sky_avg = sum(self.of_det_sky)/float(window_size)

            threshold = 1.5

            if val_losses[0] > threshold*gnd_avg:
                self.gnd_dw = True

            if val_losses[1] > threshold*obj_avg:
                self.obj_dw = True

            if val_losses[2] > threshold*bld_avg:
                self.bld_dw = True

            if val_losses[3] > threshold*veg_avg:
                self.veg_dw = True

            if val_losses[4] > threshold*sky_avg:
                self.sky_dw = True

            self.of_det_gnd.popleft()
            self.of_det_obj.popleft()
            self.of_det_bld.popleft()
            self.of_det_veg.popleft()
            self.of_det_sky.popleft()

            self.of_det_gnd.append(val_losses[0])
            self.of_det_obj.append(val_losses[1])
            self.of_det_bld.append(val_losses[2])
            self.of_det_veg.append(val_losses[3])
            self.of_det_sky.append(val_losses[4])

    def cost_definition(self,output,label_series):

        self.c_w1 = tf.placeholder('float')
        self.c_w2 = tf.placeholder('float')
        self.c_w3 = tf.placeholder('float')
        self.c_w4 = tf.placeholder('float')
        self.c_w5 = tf.placeholder('float')


        cost = tf.nn.l2_loss(label_series[0][-1]-output[0]) + \
               tf.nn.l2_loss(label_series[1][-1]-output[1]) + \
               tf.nn.l2_loss(label_series[2][-1]-output[2]) + \
               tf.nn.l2_loss(tf.multiply(label_series[4][-1],label_series[3][-1])-tf.multiply(label_series[4][-1],output[3])) + \
               self.c_w1*tf.nn.l2_loss(label_series[5][-1]-output[4]) + \
               self.c_w2*tf.nn.l2_loss(label_series[6][-1]-output[5]) + \
               self.c_w3*tf.nn.l2_loss(label_series[7][-1]-output[6]) + \
               self.c_w4*tf.nn.l2_loss(label_series[8][-1]-output[7]) + \
               self.c_w5*tf.nn.l2_loss(label_series[9][-1]-output[8])

        loss = tf.nn.l2_loss(label_series[0][-1]-output[0]) + \
               tf.nn.l2_loss(label_series[1][-1]-output[1]) + \
               tf.nn.l2_loss(label_series[2][-1]-output[2]) + \
               tf.nn.l2_loss(tf.multiply(label_series[4][-1],label_series[3][-1])-tf.multiply(label_series[4][-1],output[3])) + \
               self.c_w1*tf.nn.l2_loss(label_series[5][-1]-output[4]) + \
               self.c_w2*tf.nn.l2_loss(label_series[6][-1]-output[5]) + \
               self.c_w3*tf.nn.l2_loss(label_series[7][-1]-output[6]) + \
               self.c_w4*tf.nn.l2_loss(label_series[8][-1]-output[7]) + \
               self.c_w5*tf.nn.l2_loss(label_series[9][-1]-output[8])



        self.collect_variables()

        regularizer = tf.contrib.layers.l2_regularizer(scale=0.0005)
        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)

        # comment when using current LSTM implementation
        rnn_reg_term = tf.contrib.layers.apply_regularization(regularizer,self.rnn_variables)

        cost += reg_term
        # comment when using current LSTM implementation
        cost += rnn_reg_term

        return cost, loss

    def split_label_series(self):

        imr = tf.unstack(self.imr_label,axis=1)
        img = tf.unstack(self.img_label,axis=1)
        imb = tf.unstack(self.imb_label,axis=1)
        depth = tf.unstack(self.depth_label,axis=1)
        depth_mask = tf.unstack(self.depth_mask,axis=1)
        gnd = tf.unstack(self.gnd_label,axis=1)
        obj = tf.unstack(self.obj_label,axis=1)
        bld = tf.unstack(self.bld_label,axis=1)
        veg = tf.unstack(self.veg_label,axis=1)
        sky = tf.unstack(self.sky_label,axis=1)

        label_series = [imr,img,imb,depth,depth_mask,gnd,obj,bld,veg,sky]

        return label_series

    def train_model(self):

        input = [self.imr_input,self.img_input,self.imb_input,self.depth_input,
                 self.gnd_input,self.obj_input,self.bld_input,self.veg_input,self.sky_input]

        output = self.network(input)

        label_series = self.split_label_series()

        cost, loss = self.cost_definition(output,label_series)

        c_w1 = 1.0
        c_w2 = 1.0
        c_w3 = 0.1
        c_w4 = 1.0
        c_w5 = 1.0


        # comment when using LSTM
        rnn_weight_norms = []
        for i in self.rnn_weights_H:
            rnn_weight_norms.append(tf.norm(i,ord='euclidean'))

        # comment when using LSTM
        rnn_weight_sum = []
        for i in range(0,len(rnn_weight_norms)):
            name = 'RNN Weight Norm ' + str(i)
            rnn_weight_sum.append(tf.summary.scalar(name,rnn_weight_norms[i]))


        normalization = tf.placeholder('float')

        epoch_loss = tf.Variable(0.0,name='epoch_loss',trainable=False)
        val_loss = tf.Variable(0.0,name='val_loss',trainable=False)
        rms = tf.Variable(0.0,name='rms_error',trainable=False)
        rel = tf.Variable(0.0,name='rel_error',trainable=False)


        imr_loss = tf.Variable(0.0,name='imr_val_loss',trainable=False)
        img_loss = tf.Variable(0.0,name='img_val_loss',trainable=False)
        imb_loss = tf.Variable(0.0,name='imb_val_loss',trainable=False)
        gnd_loss = tf.Variable(0.0,name='gnd_val_loss',trainable=False)
        obj_loss = tf.Variable(0.0,name='obj_val_loss',trainable=False)
        bld_loss = tf.Variable(0.0,name='bld_val_loss',trainable=False)
        veg_loss = tf.Variable(0.0,name='veg_val_loss',trainable=False)
        sky_loss = tf.Variable(0.0,name='sky_val_loss',trainable=False)

        imr_loss_reset = imr_loss.assign(0.0)
        img_loss_reset = img_loss.assign(0.0)
        imb_loss_reset = imb_loss.assign(0.0)
        gnd_loss_reset = gnd_loss.assign(0.0)
        obj_loss_reset = obj_loss.assign(0.0)
        bld_loss_reset = bld_loss.assign(0.0)
        veg_loss_reset = veg_loss.assign(0.0)
        sky_loss_reset = sky_loss.assign(0.0)

        imr_loss_update = imr_loss.assign_add(tf.nn.l2_loss(label_series[0][-1]-output[0])/normalization)
        img_loss_update = img_loss.assign_add(tf.nn.l2_loss(label_series[1][-1]-output[1])/normalization)
        imb_loss_update = imb_loss.assign_add(tf.nn.l2_loss(label_series[2][-1]-output[2])/normalization)
        gnd_loss_update = gnd_loss.assign_add(tf.nn.l2_loss(label_series[5][-1]-output[4])/normalization)
        obj_loss_update = obj_loss.assign_add(tf.nn.l2_loss(label_series[6][-1]-output[5])/normalization)
        bld_loss_update = bld_loss.assign_add(tf.nn.l2_loss(label_series[7][-1]-output[6])/normalization)
        veg_loss_update = veg_loss.assign_add(tf.nn.l2_loss(label_series[8][-1]-output[7])/normalization)
        sky_loss_update = sky_loss.assign_add(tf.nn.l2_loss(label_series[9][-1]-output[8])/normalization)

        summary_imr = tf.summary.scalar('Red Channel Validation Loss', imr_loss)
        summary_img = tf.summary.scalar('Green Channel Validation Loss', img_loss)
        summary_imb = tf.summary.scalar('Blue Channel Validation Loss', imb_loss)
        summary_gnd = tf.summary.scalar('Ground Channel Validation Loss', gnd_loss)
        summary_obj = tf.summary.scalar('Object Channel Validation Loss', obj_loss)
        summary_bld = tf.summary.scalar('Building Channel Validation Loss', bld_loss)
        summary_veg = tf.summary.scalar('Vegetation Channel Validation Loss', veg_loss)
        summary_sky = tf.summary.scalar('Sky Channel Validation Loss', sky_loss)

        epoch_loss_reset = epoch_loss.assign(0)
        epoch_loss_update = epoch_loss.assign_add(cost)

        rms_plh = tf.placeholder('float')
        rel_plh = tf.placeholder('float')

        loss_val_reset = val_loss.assign(0)
        loss_val_update = val_loss.assign_add(loss/normalization)

        rms_reset = rms.assign(0.0)
        rms_update = rms.assign_add(rms_plh)

        rel_reset = rel.assign(0.0)
        rel_update = rel.assign_add(rel_plh)

        sum_epoch_loss = tf.summary.scalar('Epoch Loss Full Model',epoch_loss)
        sum_val_loss = tf.summary.scalar('Validation Loss Full Model',val_loss)
        summary_rms = tf.summary.scalar('RMS Error in Validation', rms)
        summary_rel = tf.summary.scalar('Relative Error in Validation', rel)

        self.training_cost = cost

        # optimizer for draft LSTM network
        #optimizer1 = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)
        #optimizer2 = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)
        #optimizer3 = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)
        #optimizer4 = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)

        optimizer1 = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.training_cost,var_list=self.rnn_variables)
        optimizer2 = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.training_cost,var_list=self.rnn_variables)
        optimizer3 = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.training_cost,var_list=self.rnn_variables+self.decoder_variables)
        optimizer4 = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.training_cost,var_list=self.rnn_variables+self.decoder_variables+self.encoder_variables)

        validations = np.arange(0, self.n_training_validations)
        set_val = np.random.choice(validations,self.n_training_validations,replace=False)

        load_MAE = tf.train.Saver({'red_ec_layer_weights':self.red_ec_layer['weights'],
                                   'red_ec_layer_bias':self.red_ec_layer['bias'],
                                   'red_dc_layer_weights':self.red_dc_layer['weights'],
                                   'red_dc_layer_bias':self.red_dc_layer['bias'],
                                   'green_ec_layer_weights':self.green_ec_layer['weights'],
                                   'green_ec_layer_bias':self.green_ec_layer['bias'],
                                   'green_dc_layer_weights':self.green_dc_layer['weights'],
                                   'green_dc_layer_bias':self.green_dc_layer['bias'],
                                   'blue_ec_layer_weights':self.blue_ec_layer['weights'],
                                   'blue_ec_layer_bias':self.blue_ec_layer['bias'],
                                   'blue_dc_layer_weights':self.blue_dc_layer['weights'],
                                   'blue_dc_layer_bias':self.blue_dc_layer['bias'],
                                   'depth_ec_layer_weights':self.depth_ec_layer['weights'],
                                   'depth_ec_layer_bias':self.depth_ec_layer['bias'],
                                   'depth_dc_layer_weights':self.depth_dc_layer['weights'],
                                   'depth_dc_layer_bias':self.depth_dc_layer['bias'],
                                   'gnd_ec_layer_weights':self.gnd_ec_layer['weights'],
                                   'gnd_ec_layer_bias':self.gnd_ec_layer['bias'],
                                   'obj_ec_layer_weights':self.obj_ec_layer['weights'],
                                   'obj_ec_layer_bias':self.obj_ec_layer['bias'],
                                   'bld_ec_layer_weights':self.bld_ec_layer['weights'],
                                   'bld_ec_layer_bias':self.bld_ec_layer['bias'],
                                   'veg_ec_layer_weights':self.veg_ec_layer['weights'],
                                   'veg_ec_layer_bias':self.veg_ec_layer['bias'],
                                   'sky_ec_layer_weights':self.sky_ec_layer['weights'],
                                   'sky_ec_layer_bias':self.sky_ec_layer['bias'],
                                   'sem_ec_layer_weights':self.sem_ec_layer['weights'],
                                   'sem_ec_layer_bias':self.sem_ec_layer['bias'],
                                   'gnd_dc_layer_weights':self.gnd_dc_layer['weights'],
                                   'gnd_dc_layer_bias':self.gnd_dc_layer['bias'],
                                   'obj_dc_layer_weights':self.obj_dc_layer['weights'],
                                   'obj_dc_layer_bias':self.obj_dc_layer['bias'],
                                   'bld_dc_layer_weights':self.bld_dc_layer['weights'],
                                   'bld_dc_layer_bias':self.bld_dc_layer['bias'],
                                   'veg_dc_layer_weights':self.veg_dc_layer['weights'],
                                   'veg_dc_layer_bias':self.veg_dc_layer['bias'],
                                   'sky_dc_layer_weights':self.sky_dc_layer['weights'],
                                   'sky_dc_layer_bias':self.sky_dc_layer['bias'],
                                   'sem_dc_layer_weights':self.sem_dc_layer['weights'],
                                   'sem_dc_layer_bias':self.sem_dc_layer['bias'],
                                   'full_ec_layer_weights':self.full_ec_layer['weights'],
                                   'full_ec_layer_bias':self.full_ec_layer['bias'],
                                   'full_dc_layer_weights':self.full_dc_layer['weights'],
                                   'full_dc_layer_bias':self.full_dc_layer['bias']})

        saver = tf.train.Saver()

        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        
        with tf.Session(config=config) as sess:

            self.flag_is_running = True

            train_writer1 = tf.summary.FileWriter(self.FLAGS.logs_dir,sess.graph)
            sess.run(tf.global_variables_initializer())

            load_MAE.restore(sess,'models/full/FullMAE/fullmodel.ckpt')

            tf.get_default_graph().finalize()

            print('----------------------------------------------------------------')
            print('Zero Validation')

            sess.run(loss_val_reset)

            norm = 8*1080*set_val.shape[0]
            error_rms = 0
            error_rel = 0

            for i in set_val:

                red_label = self.imr_val[i]
                green_label = self.img_val[i]
                blue_label = self.imb_val[i]
                depth_label = self.depth_val[i]
                depth_mask = self.depth_mask_val[i]
                gnd_label = self.gnd_val[i]
                obj_label = self.obj_val[i]
                bld_label = self.bld_val[i]
                veg_label = self.veg_val[i]
                sky_label = self.sky_val[i]

                norm += np.count_nonzero(depth_mask[-1])

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
                                                                                                    rnn=True,
                                                                                                    singleframe=True)

                feed_dict = {self.imr_input:imr_in,
                             self.img_input:img_in,
                             self.imb_input:imb_in,
                             self.depth_input:depth_in,
                             self.gnd_input:gnd_in,
                             self.obj_input:obj_in,
                             self.bld_input:bld_in,
                             self.veg_input:veg_in,
                             self.sky_input:sky_in,
                             self.depth_mask:[depth_mask],
                             self.imr_label:[red_label],
                             self.img_label:[green_label],
                             self.imb_label:[blue_label],
                             self.depth_label:[depth_label],
                             self.gnd_label:[gnd_label],
                             self.obj_label:[obj_label],
                             self.bld_label:[bld_label],
                             self.veg_label:[veg_label],
                             self.sky_label:[sky_label],
                             self.init_states:np.zeros((1,self.state_size)),
                             normalization:norm}

                cost_dict = {self.c_w1:c_w1,
                             self.c_w2:c_w2,
                             self.c_w3:c_w3,
                             self.c_w4:c_w4,
                             self.c_w5:c_w5}

                feed_dict.update(cost_dict)

                im_pred,c_val = sess.run([output,loss_val_update],feed_dict=feed_dict)

                depth_pred = BR.invert_depth(im_pred[3])
                depth_gt = BR.invert_depth(depth_label[-1])

                error_rms += eval.rms_error(depth_pred,depth_gt)
                error_rel += eval.relative_error(depth_pred,depth_gt)

            sum_val = sess.run(sum_val_loss)
            train_writer1.add_summary(sum_val,0)
            print('Validation Loss (per pixel): ', sess.run(val_loss.value()))
            print('RMSE Error over Validation Set:', error_rms/self.n_training_validations)
            print('Relative Error over Validation Set:', error_rel/self.n_training_validations)
            print('-----------------------------------------------------------------')




            for epoch in range(0,self.hm_epochs):
                sess.run(epoch_loss_reset)
                time1 = datetime.now()

                in_state = np.zeros((self.batch_size,self.state_size))


                for batch in range(0,self.n_batches):



                    batch_sequences = self.train_sequences[batch*self.batch_size:(batch+1)*self.batch_size]
                    frames = BR.get_frames(batch_sequences,self.data_train,size_input=1080)

                    imr_batch_label = frames[0]
                    img_batch_label = frames[1]
                    imb_batch_label = frames[2]
                    depth_batch_label = frames[3]
                    depth_mask_batch = frames[4]
                    gnd_batch_label = frames[5]
                    obj_batch_label = frames[6]
                    bld_batch_label = frames[7]
                    veg_batch_label = frames[8]
                    sky_batch_label = frames[9]

                    imr_in,img_in,imb_in,depth_in,gnd_in,obj_in,bld_in,veg_in,sky_in = input_distortion(copy(frames[0]),
                                                                                                         copy(frames[1]),
                                                                                                         copy(frames[2]),
                                                                                                         copy(frames[3]),
                                                                                                         copy(frames[5]),
                                                                                                         copy(frames[6]),
                                                                                                         copy(frames[7]),
                                                                                                         copy(frames[8]),
                                                                                                         copy(frames[9]),
                                                                                                         resolution=(18,60),
                                                                                                         rnn=True)

                    imr_in,img_in,imb_in,depth_in,gnd_in,obj_in,bld_in,veg_in,sky_in = self.augment_data(imr_in,
                                                                                                         img_in,
                                                                                                         imb_in,
                                                                                                         depth_in,
                                                                                                         gnd_in,
                                                                                                         obj_in,
                                                                                                         bld_in,
                                                                                                         veg_in,
                                                                                                         sky_in)


                    # horizontal mirroring
                    ind_batch = np.linspace(0,self.batch_size-1,self.batch_size).astype(int)
                    ind_rand_who = np.random.choice(ind_batch,int(self.batch_size/2),replace=False)




                    imr_in = BR.horizontal_mirroring(imr_in,ind_rand_who,option='RNN')
                    imr_batch_label = BR.horizontal_mirroring(imr_batch_label,ind_rand_who,option='RNN')
                    img_in = BR.horizontal_mirroring(img_in,ind_rand_who,option='RNN')
                    img_batch_label = BR.horizontal_mirroring(img_batch_label,ind_rand_who,option='RNN')
                    imb_in = BR.horizontal_mirroring(imb_in,ind_rand_who,option='RNN')
                    imb_batch_label = BR.horizontal_mirroring(imb_batch_label,ind_rand_who,option='RNN')

                    depth_in = BR.horizontal_mirroring(depth_in,ind_rand_who,option='RNN')
                    depth_batch_label = BR.horizontal_mirroring(depth_batch_label,ind_rand_who,option='RNN')
                    depth_mask_batch = BR.horizontal_mirroring(depth_mask_batch,ind_rand_who,option='RNN')

                    gnd_in = BR.horizontal_mirroring(gnd_in,ind_rand_who,option='RNN')
                    gnd_batch_label = BR.horizontal_mirroring(gnd_batch_label,ind_rand_who,option='RNN')
                    obj_in = BR.horizontal_mirroring(obj_in,ind_rand_who,option='RNN')
                    obj_batch_label = BR.horizontal_mirroring(obj_batch_label,ind_rand_who,option='RNN')
                    bld_in = BR.horizontal_mirroring(bld_in,ind_rand_who,option='RNN')
                    bld_batch_label = BR.horizontal_mirroring(bld_batch_label,ind_rand_who,option='RNN')
                    veg_in = BR.horizontal_mirroring(veg_in,ind_rand_who,option='RNN')
                    veg_batch_label = BR.horizontal_mirroring(veg_batch_label,ind_rand_who,option='RNN')
                    sky_in = BR.horizontal_mirroring(sky_in,ind_rand_who,option='RNN')
                    sky_batch_label = BR.horizontal_mirroring(sky_batch_label,ind_rand_who,option='RNN')


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
                                 self.imr_label:imr_batch_label,
                                 self.img_label:img_batch_label,
                                 self.imb_label:imb_batch_label,
                                 self.depth_label:depth_batch_label,
                                 self.gnd_label:gnd_batch_label,
                                 self.obj_label:obj_batch_label,
                                 self.bld_label:bld_batch_label,
                                 self.veg_label:veg_batch_label,
                                 self.sky_label:sky_batch_label,
                                 self.init_states:in_state}

                    cost_dict = {self.c_w1:c_w1,
                                 self.c_w2:c_w2,
                                 self.c_w3:c_w3,
                                 self.c_w4:c_w4,
                                 self.c_w5:c_w5}

                    feed_dict.update(cost_dict)

                    # training operation (first only full encoding is trained, then (after 10 epochs) everything is trained
                    if epoch < 10:
                        _ , c, l  = sess.run([optimizer1, cost, epoch_loss_update], feed_dict=feed_dict)

                    if epoch >= 10 and epoch < 30:
                        _ , c, l = sess.run([optimizer2, cost, epoch_loss_update], feed_dict=feed_dict)

                    if epoch >= 30 and epoch < 60:
                        _ , c, l = sess.run([optimizer3, cost, epoch_loss_update], feed_dict=feed_dict)

                    else:
                        _ , c, l = sess.run([optimizer4, cost, epoch_loss_update], feed_dict=feed_dict)

                sum_train = sess.run(sum_epoch_loss)
                train_writer1.add_summary(sum_train,epoch)

                for i in rnn_weight_sum:
                    sum = sess.run(i)
                    train_writer1.add_summary(sum,epoch)

                print('----------------------------------------------------------------')
                print('Epoch', epoch, 'completed out of', self.hm_epochs)
                print('Training Loss (per epoch): ', sess.run(epoch_loss.value()))



                sess.run([loss_val_reset,
                          rms_reset,
                          rel_reset,
                          imr_loss_reset,
                          img_loss_reset,
                          imb_loss_reset,
                          gnd_loss_reset,
                          obj_loss_reset,
                          bld_loss_reset,
                          veg_loss_reset,
                          sky_loss_reset])

                norm = 8*1080*set_val.shape[0]




                error_rms = 0
                error_rel = 0



                for i in set_val:

                    red_label = self.imr_val[i]
                    green_label = self.img_val[i]
                    blue_label = self.imb_val[i]
                    depth_label = self.depth_val[i]
                    depth_mask = self.depth_mask_val[i]
                    gnd_label = self.gnd_val[i]
                    obj_label = self.obj_val[i]
                    bld_label = self.bld_val[i]
                    veg_label = self.veg_val[i]
                    sky_label = self.sky_val[i]


                    norm += np.count_nonzero(depth_mask[-1])

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
                                                                                                        rnn=True,
                                                                                                        singleframe=True)


                    feed_dict = {self.imr_input:imr_in,
                                 self.img_input:img_in,
                                 self.imb_input:imb_in,
                                 self.depth_input:depth_in,
                                 self.gnd_input:gnd_in,
                                 self.obj_input:obj_in,
                                 self.bld_input:bld_in,
                                 self.veg_input:veg_in,
                                 self.sky_input:sky_in,
                                 self.depth_mask:[depth_mask],
                                 self.imr_label:[red_label],
                                 self.img_label:[green_label],
                                 self.imb_label:[blue_label],
                                 self.depth_label:[depth_label],
                                 self.gnd_label:[gnd_label],
                                 self.obj_label:[obj_label],
                                 self.bld_label:[bld_label],
                                 self.veg_label:[veg_label],
                                 self.sky_label:[sky_label],
                                 self.init_states:np.zeros((1,self.state_size)),
                                 normalization:norm}

                    cost_dict = {self.c_w1:c_w1,
                                 self.c_w2:c_w2,
                                 self.c_w3:c_w3,
                                 self.c_w4:c_w4,
                                 self.c_w5:c_w5}

                    feed_dict.update(cost_dict)

                    im_pred,c_val,l_imr,l_img,l_imb,l_gnd,l_obj,l_bld,l_veg,l_sky = sess.run([output,
                                                                                              loss_val_update,
                                                                                              imr_loss_update,
                                                                                              img_loss_update,
                                                                                              imb_loss_update,
                                                                                              gnd_loss_update,
                                                                                              obj_loss_update,
                                                                                              bld_loss_update,
                                                                                              veg_loss_update,
                                                                                              sky_loss_update],feed_dict=feed_dict)

                    depth_pred = BR.invert_depth(im_pred[3])
                    depth_gt = BR.invert_depth(depth_label[-1])

                    error_rms += eval.rms_error(depth_pred,depth_gt)
                    error_rel += eval.relative_error(depth_pred,depth_gt)


                error_rms = error_rms/self.n_training_validations
                error_rel = error_rel/self.n_training_validations

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

                # test for overfitting
                if epoch >= 30:
                    g,o,b,v,s = sess.run([gnd_loss.value(),obj_loss.value(),bld_loss.value(),veg_loss.value(),sky_loss.value()])
                    val_losses = [g,o,b,v,s]
                    self.overfitting_detection(val_losses,epoch)

                    if self.gnd_dw:
                        c_w1 = 0.1
                    if self.obj_dw:
                        c_w2 = 0.1
                    if self.bld_dw:
                        c_w3 = 0.1
                    if self.veg_dw:
                        c_w4 = 0.1
                    if self.obj_dw:
                        c_w5 = 0.1



                print('Validation Loss (per pixel): ', sess.run(val_loss.value()))
                print('RMSE Error over Validation Set:', sess.run(rms.value()))
                print('Relative Error over Validation Set:',sess.run(rel.value()))

                time2 = datetime.now()
                delta = time2-time1
                print('Epoch Time [seconds]:', delta.seconds)
                print('-----------------------------------------------------------------')

            if self.saving == True:
                saver.save(sess,self.FLAGS.model_dir+'/fullmodel_rnn.ckpt')
                print('SAVED MODEL')

    def evaluate(self,data_test,run=False):

        self.data_test = data_test
        self.prepare_test_data()

        encoding = self.encoding_network(self.imr_input,
                                         self.img_input,
                                         self.imb_input,
                                         self.depth_input,
                                         self.gnd_input,
                                         self.obj_input,
                                         self.bld_input,
                                         self.veg_input,
                                         self.sky_input)

        outputs, _current_state = self.Basic_RNN(encoding)
        output = self.decoding_network(outputs)


        load_weights = tf.train.Saver()

        if run==False:
            raise ValueError

        dir = 'models/rnn/' + run

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            load_weights.restore(sess,dir+'/fullmodel.ckpt') # runs from 06112017 it ist fullmodel_rnn

            n_evaluations = 697
            print('Size of Test Set:',n_evaluations)

            error_rms = 0
            error_rel = 0

            in_state = np.zeros((1,self.state_size))

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



                imr_in,img_in,imb_in,depth_in,gnd_in,obj_in,bld_in,veg_in,sky_in = input_distortion(copy(imr_label),
                                                                                                    copy(img_label),
                                                                                                    copy(imb_label),
                                                                                                    copy(depth_label),
                                                                                                    copy(gnd_label),
                                                                                                    copy(obj_label),
                                                                                                    copy(bld_label),
                                                                                                    copy(veg_label),
                                                                                                    copy(sky_label),
                                                                                                    resolution=(18,60),
                                                                                                    rnn=True,
                                                                                                    singleframe=True)


                feed_dict = {self.imr_input:imr_in,
                             self.img_input:img_in,
                             self.imb_input:imb_in,
                             self.depth_input:depth_in,
                             self.gnd_input:gnd_in,
                             self.obj_input:obj_in,
                             self.bld_input:bld_in,
                             self.veg_input:veg_in,
                             self.sky_input:sky_in,
                             self.init_states:in_state}

                pred = sess.run(output,feed_dict=feed_dict)
                depth_pred = pred[3]


                inv_depth_pred = np.asarray(depth_pred)
                inv_depth_label = np.asarray(depth_label[-1])

                gt = BR.invert_depth(inv_depth_label)
                est = BR.invert_depth(inv_depth_pred)

                error_rms += eval.rms_error(est,gt)
                error_rel += eval.relative_error(est,gt)


            print('Error (RMS):', error_rms/n_evaluations)
            print('Error (REL):', error_rel/n_evaluations)

# load data

data_train = load_training_data()
data_validate = load_validation_data()

# running model

rnn_mae = RecurrentMAE(data_train,data_validate)
rnn_mae.train_model()

#rnn_mae.evaluate(run='20171103-074407')





















