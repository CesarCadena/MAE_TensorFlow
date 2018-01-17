# model based on 'Multi-modal Auto-Encoders as Joint Estimators for Robotics Scene Understanding' by Cadena et al.
# code developed by Silvan Weder
from collections import deque
from datetime import datetime
from copy import copy, deepcopy

import tensorflow as tf
import numpy as np

import os
import json

import evaluation_functions as eval
import basic_routines as BR
from input_distortion import input_distortion
from models import RNN_MAE, full_MAE

from build_test_sequences import build_test_sequences,distort_test_sequences


# deleted blue_dc_layer_bias from loader (wrong name in full model)
# need to change that afterwards, when corrected full model is trained



class RecurrentMAE:

    def __init__(self,n_epochs=None,rnn_option='basic',n_rnn_steps=5,mirroring=False, learning_rate=None,
                 load_previous=False,resolution=(18,60),sharing='nonshared',model='new'):

        if n_epochs == None:
            raise ValueError('no number of epochs passed')

        if learning_rate == None:
            raise ValueError('no learning rate given')


        self.height = resolution[0]
        self.width = resolution[1]

        self.size_input = self.height*self.width
        self.size_coding = 1024

        self.n_training_validations = 200

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

        self.model = model

         # recurrent options

        self.n_rnn_steps = n_rnn_steps
        self.state_size = 1024
        self.rnn_option = rnn_option
        self.sharing = sharing


        # prepare data
        self.data_augmentation = True
        self.n_augmentations = 3

        # mirroring
        self.mirroring = mirroring

        # flags
        self.flag_is_running = False

        self.placeholder_definition()


        self.learning_rate = learning_rate
        self.hm_epochs = n_epochs


        # variables for overfitting detection
        self.of_det_imr = deque([])
        self.of_det_img = deque([])
        self.of_det_imb = deque([])
        self.of_det_gnd = deque([])
        self.of_det_obj = deque([])
        self.of_det_bld = deque([])
        self.of_det_veg = deque([])
        self.of_det_sky = deque([])

        self.imr_dw = False
        self.img_dw = False
        self.imb_dw = False
        self.gnd_dw = False
        self.obj_dw = False
        self.bld_dw = False
        self.veg_dw = False
        self.sky_dw = False

        self.imr_avg_min = np.infty
        self.img_avg_min = np.infty
        self.imb_avg_min = np.infty
        self.gnd_avg_min = np.infty
        self.obj_avg_min = np.infty
        self.bld_avg_min = np.infty
        self.veg_avg_min = np.infty
        self.sky_avg_min = np.infty

        # model savings
        self.saving = True
        now = datetime.now()

        # model loading
        self.load_previous = load_previous

        self.folder_model = 'models/'
        self.folder_logs = 'logs/'

        self.mode = 'rnn/'
        self.run = now.strftime('%Y%m%d-%H%M%S')

        self.project_dir = './'
        self.model_dir = self.project_dir + self.folder_model + self.mode + self.run
        self.logs_dir = self.project_dir + self.folder_logs + self.mode + self.run

        if load_previous == False:
            self.load_dir = self.project_dir + self.folder_model + 'full/FullMAE1/'
        if load_previous == True:
            self.load_dir = self.project_dir + self.folder_model + self.mode + 'previous/'

        os.mkdir(self.model_dir)
        os.mkdir(self.logs_dir)

        self.specifications = {'mode': self.rnn_option,
                               'number of rnn steps': self.n_rnn_steps,
                               'mirroring': str(self.mirroring),
                               'learning rate': self.learning_rate,
                               'number of epochs': self.hm_epochs,
                               'sharing':self.sharing}

        json.dump(self.specifications, open(self.logs_dir+"/specs.txt",'w'))

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

        # learning rate placeholder
        self.lr = tf.placeholder('float')

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

            u = np.random.randint(0,10)

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

                if u == 9:

                    if rnn_step%2 == 0:

                        depth_series_aug[rnn_step] = zeros
                        gnd_series_aug[rnn_step] = zeros
                        obj_series_aug[rnn_step] = zeros
                        bld_series_aug[rnn_step] = zeros
                        veg_series_aug[rnn_step] = zeros
                        sky_series_aug[rnn_step] = zeros

                    else:
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
        self.depth_sparse_test = []
        self.gnd_test = []
        self.obj_test = []
        self.bld_test = []
        self.veg_test = []
        self.sky_test = []


        for i in self.data_test:

            imr_series = []
            img_series = []
            imb_series = []
            depth_series = []
            depth_sparse_series = []
            depth_mask_series = []
            gnd_series = []
            obj_series = []
            bld_series = []
            veg_series = []
            sky_series = []

            for j in range(len(i)):

                imr_series.append(i[j]['xcr']/255.)
                img_series.append(i[j]['xcg']/255.)
                imb_series.append(i[j]['xcb']/255.)
                depth_series.append(i[j]['xid'])
                depth_sparse_series.append(i[j]['xidsparse'])
                depth_mask_series.append(i[j]['xmask'])
                gnd_series.append((i[j]['sem']==1).astype(int))
                obj_series.append((i[j]['sem']==2).astype(int))
                bld_series.append((i[j]['sem']==3).astype(int))
                veg_series.append((i[j]['sem']==4).astype(int))
                sky_series.append((i[j]['sem']==5).astype(int))


            self.imr_test.append(copy(imr_series))
            self.img_test.append(copy(img_series))
            self.imb_test.append(copy(imb_series))
            self.depth_test.append(copy(depth_series))
            self.depth_sparse_test.append(copy(depth_sparse_series))
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

    def network(self, input):

        output = RNN_MAE(input[0],input[1],input[2],input[3],input[4],input[5],input[6],input[7],input[8],
                         n_rnn_steps=self.n_rnn_steps,init_states=self.init_states,option=self.rnn_option,sharing=self.sharing)
        return output

    def overfitting_detection(self,val_losses):

        window_size = 10

        if len(self.of_det_gnd) < window_size:
            self.of_det_imr.append(val_losses[0])
            self.of_det_img.append(val_losses[1])
            self.of_det_imb.append(val_losses[2])
            self.of_det_gnd.append(val_losses[3])
            self.of_det_obj.append(val_losses[4])
            self.of_det_bld.append(val_losses[5])
            self.of_det_veg.append(val_losses[6])
            self.of_det_sky.append(val_losses[7])

        else:
            imr_avg = sum(self.of_det_imr)/float(window_size)
            img_avg = sum(self.of_det_img)/float(window_size)
            imb_avg = sum(self.of_det_imb)/float(window_size)
            gnd_avg = sum(self.of_det_gnd)/float(window_size)
            obj_avg = sum(self.of_det_obj)/float(window_size)
            bld_avg = sum(self.of_det_bld)/float(window_size)
            veg_avg = sum(self.of_det_veg)/float(window_size)
            sky_avg = sum(self.of_det_sky)/float(window_size)


            threshold = 1.2

            if imr_avg > threshold*self.imr_avg_min:
                self.imr_dw = True
            else:
                if imr_avg < self.imr_avg_min:
                    self.imr_avg_min = imr_avg

            if img_avg > threshold*self.img_avg_min:
                self.img_dw = True
            else:
                if img_avg < self.img_avg_min:
                    self.img_avg_min = img_avg

            if imb_avg > threshold*self.imb_avg_min:
                self.imb_dw = True
            else:
                if imb_avg < self.imb_avg_min:
                    self.imb_avg_min = imb_avg

            if gnd_avg > threshold*self.gnd_avg_min:
                self.gnd_dw = True
            else:
                if gnd_avg < self.gnd_avg_min:
                    self.gnd_avg_min = gnd_avg

            if obj_avg > threshold*self.obj_avg_min:
                self.obj_dw = True
            else:
                if obj_avg < self.obj_avg_min:
                    self.obj_avg_min = obj_avg

            if bld_avg > threshold*self.bld_avg_min:
                self.bld_dw = True
            else:
                if bld_avg < self.bld_avg_min:
                    self.bld_avg_min = bld_avg

            if veg_avg > threshold*self.veg_avg_min:
                self.veg_dw = True
            else:
                if veg_avg < self.veg_avg_min:
                    self.veg_avg_min = veg_avg

            if sky_avg > threshold*self.sky_avg_min:
                self.sky_dw = True
            else:
                if sky_avg < self.sky_avg_min:
                    self.sky_avg_min = sky_avg

            self.of_det_imr.popleft()
            self.of_det_img.popleft()
            self.of_det_imb.popleft()
            self.of_det_gnd.popleft()
            self.of_det_obj.popleft()
            self.of_det_bld.popleft()
            self.of_det_veg.popleft()
            self.of_det_sky.popleft()

            self.of_det_imr.append(val_losses[0])
            self.of_det_img.append(val_losses[1])
            self.of_det_imb.append(val_losses[2])
            self.of_det_gnd.append(val_losses[3])
            self.of_det_obj.append(val_losses[4])
            self.of_det_bld.append(val_losses[5])
            self.of_det_veg.append(val_losses[6])
            self.of_det_sky.append(val_losses[7])

    def cost_definition(self,output,label_series):

        self.c_w1 = tf.placeholder('float')
        self.c_w2 = tf.placeholder('float')
        self.c_w3 = tf.placeholder('float')
        self.c_w4 = tf.placeholder('float')
        self.c_w5 = tf.placeholder('float')

        self.c_r = tf.placeholder('float')
        self.c_g = tf.placeholder('float')
        self.c_b = tf.placeholder('float')

        if self.rnn_option == 'gated':
            cost = self.c_r*tf.nn.l2_loss(label_series[0][-1]-output[0]) + \
                   self.c_g*tf.nn.l2_loss(label_series[1][-1]-output[1]) + \
                   self.c_b*tf.nn.l2_loss(label_series[2][-1]-output[2]) + \
                   50*tf.nn.l2_loss(tf.multiply(label_series[4][-1],label_series[3][-1])-tf.multiply(label_series[4][-1],output[3])) +\
                   0.001*self.c_w1*tf.nn.l2_loss(label_series[5][-1]-output[4]) + \
                   0.001*self.c_w2*tf.nn.l2_loss(label_series[6][-1]-output[5]) + \
                   0.001*self.c_w3*tf.nn.l2_loss(label_series[7][-1]-output[6]) + \
                   0.001*self.c_w4*tf.nn.l2_loss(label_series[8][-1]-output[7]) + \
                   0.001*self.c_w5*tf.nn.l2_loss(label_series[9][-1]-output[8]) + \
                   50*tf.losses.absolute_difference(tf.multiply(label_series[4][-1],label_series[3][-1]),tf.multiply(label_series[4][-1],output[3]))

        else:
            cost = self.c_r*tf.nn.l2_loss(label_series[0][-1]-output[0]) + \
                   self.c_g*tf.nn.l2_loss(label_series[1][-1]-output[1]) + \
                   self.c_b*tf.nn.l2_loss(label_series[2][-1]-output[2]) + \
                   100*tf.nn.l2_loss(tf.multiply(label_series[4][-1],label_series[3][-1])-tf.multiply(label_series[4][-1],output[3])) +\
                   0.01*self.c_w1*tf.nn.l2_loss(label_series[5][-1]-output[4]) + \
                   0.01*self.c_w2*tf.nn.l2_loss(label_series[6][-1]-output[5]) + \
                   0.01*self.c_w3*tf.nn.l2_loss(label_series[7][-1]-output[6]) + \
                   0.01*self.c_w4*tf.nn.l2_loss(label_series[8][-1]-output[7]) + \
                   0.01*self.c_w5*tf.nn.l2_loss(label_series[9][-1]-output[8])
        #100*tf.losses.absolute_difference(tf.multiply(label_series[4][-1],label_series[3][-1]),tf.multiply(label_series[4][-1],output[3])) +\

        loss = tf.nn.l2_loss(label_series[0][-1]-output[0]) + \
               tf.nn.l2_loss(label_series[1][-1]-output[1]) + \
               tf.nn.l2_loss(label_series[2][-1]-output[2]) + \
               tf.nn.l2_loss(tf.multiply(label_series[4][-1],label_series[3][-1])-tf.multiply(label_series[4][-1],output[3])) + \
               tf.nn.l2_loss(label_series[5][-1]-output[4]) + \
               tf.nn.l2_loss(label_series[6][-1]-output[5]) + \
               tf.nn.l2_loss(label_series[7][-1]-output[6]) + \
               tf.nn.l2_loss(label_series[8][-1]-output[7]) + \
               tf.nn.l2_loss(label_series[9][-1]-output[8])

        if self.rnn_option == 'basic':
            regularizer = tf.contrib.layers.l2_regularizer(scale=0.0005) #basic rnn
        if self.rnn_option == 'lstm':
            regularizer = tf.contrib.layers.l2_regularizer(scale=0.0005) #lstm rnn
        if self.rnn_option == 'gated':
            regularizer = tf.contrib.layers.l2_regularizer(scale=0.00005) #lstm rnn

        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)

        #cost += 30*reg_term #basic rnn
        cost += 30*reg_term

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

    def train_model(self,data_train,data_validate):

        self.data_train = data_train
        self.data_validate = data_validate

        self.prepare_training_data()
        self.prepare_validation_data()

        # training options
        self.batch_size = 60
        self.n_batches = int(len(self.train_sequences)/self.batch_size)


        input = [self.imr_input,self.img_input,
                 self.imb_input,self.depth_input,
                 self.gnd_input,self.obj_input,
                 self.bld_input,self.veg_input,self.sky_input]

        output = self.network(input)

        label_series = self.split_label_series()

        cost, loss = self.cost_definition(output,label_series)

        c_wr = 1.0
        c_wg = 1.0
        c_wb = 1.0
        c_w1 = 1.0
        c_w2 = 1.0
        c_w3 = 1.0
        c_w4 = 1.0
        c_w5 = 1.0


        # get rnn variables
        self.rnn_variables_H = [v for v in tf.global_variables() if v.name.startswith('RNN/H')]
        self.rnn_variables = [v for v in tf.global_variables() if v.name.startswith('RNN')]
        self.decoder_variables = [v for v in tf.global_variables() if v.name.startswith('Decoding')]
        self.encoder_variables = [v for v in tf.global_variables() if v.name.startswith('Encoding')]

        # comment when using LSTM
        rnn_weight_norms = []
        for i in self.rnn_variables_H:
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

        global_step = tf.Variable(0,trainable=False)



        if self.rnn_option == 'lstm':
            base_rate = 1e-03 # lstm RNN
            self.learning_rate = tf.train.exponential_decay(base_rate,global_step,5000, 0.96, staircase=True) # lstm configuration

        if self.rnn_option == 'basic':
            base_rate = 1e-06 # basic RNN
            lr1 = 0.1*base_rate
            lr2 = 0.1*lr1
            lr3 = 0.1*lr2
            lr4 = 0.1*lr3
            self.learning_rate = tf.train.piecewise_constant(global_step,
                                                             [20*self.n_batches,40*self.n_batches,60*self.n_batches,80*self.n_batches,100*self.n_batches],
                                                             [base_rate,lr1,lr2,lr3,lr4])

        if self.rnn_option == 'gated':
            base_rate = 1e-04
            self.learning_rate = tf.train.exponential_decay(base_rate,global_step,1000, 0.96, staircase=True) # GRU configuration



        summary_lr = tf.summary.scalar('Learning Rate',self.learning_rate)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        gvs0 = optimizer.compute_gradients(self.training_cost,var_list=self.rnn_variables)
        gvs1 = gvs0
        gvs2 = optimizer.compute_gradients(self.training_cost,var_list=self.rnn_variables+self.decoder_variables)
        gvs3 = optimizer.compute_gradients(self.training_cost,var_list=self.rnn_variables+self.decoder_variables+self.encoder_variables)

        capped_gvs0 = [(tf.clip_by_norm(grad,2), var) for grad, var in gvs0]
        capped_gvs1 = [(tf.clip_by_norm(grad,2), var) for grad, var in gvs1]
        capped_gvs2 = [(tf.clip_by_norm(grad,2), var) for grad, var in gvs2]
        capped_gvs3 = [(tf.clip_by_norm(grad,2), var) for grad, var in gvs3]

        train_op0 = optimizer.apply_gradients(capped_gvs0,global_step=global_step)
        train_op1 = optimizer.apply_gradients(capped_gvs1,global_step=global_step)
        train_op2 = optimizer.apply_gradients(capped_gvs2,global_step=global_step)
        train_op3 = optimizer.apply_gradients(capped_gvs3,global_step=global_step)


        validations = np.arange(0, self.n_training_validations)
        set_val = np.random.choice(validations,self.n_training_validations,replace=False)

        if self.model == 'old' and self.load_previous == False:

            with tf.variable_scope('Encoding',reuse=True):

                load_ec_MAE = tf.train.Saver({'red_ec_layer_weights':tf.get_variable('red_ec_layer_weights'),
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

                load_dc_MAE = tf.train.Saver({'red_dc_layer_weights':tf.get_variable('red_dc_layer_weights'),
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

        else:
            with tf.variable_scope('Encoding',reuse=True):

                load_ec_MAE = tf.train.Saver({'Encoding/red_ec_layer_weights':tf.get_variable('red_ec_layer_weights'),
                                              'Encoding/red_ec_layer_bias':tf.get_variable('red_ec_layer_bias'),
                                              'Encoding/green_ec_layer_weights':tf.get_variable('green_ec_layer_weights'),
                                              'Encoding/green_ec_layer_bias':tf.get_variable('green_ec_layer_bias'),
                                              'Encoding/blue_ec_layer_weights':tf.get_variable('blue_ec_layer_weights'),
                                              'Encoding/blue_ec_layer_bias':tf.get_variable('blue_ec_layer_bias'),
                                              'Encoding/depth_ec_layer_weights':tf.get_variable('depth_ec_layer_weights'),
                                              'Encoding/depth_ec_layer_bias':tf.get_variable('depth_ec_layer_bias'),
                                              'Encoding/gnd_ec_layer_weights':tf.get_variable('gnd_ec_layer_weights'),
                                              'Encoding/gnd_ec_layer_bias':tf.get_variable('gnd_ec_layer_bias'),
                                              'Encoding/obj_ec_layer_weights':tf.get_variable('obj_ec_layer_weights'),
                                              'Encoding/obj_ec_layer_bias':tf.get_variable('obj_ec_layer_bias'),
                                              'Encoding/bld_ec_layer_weights':tf.get_variable('bld_ec_layer_weights'),
                                              'Encoding/bld_ec_layer_bias':tf.get_variable('bld_ec_layer_bias'),
                                              'Encoding/veg_ec_layer_weights':tf.get_variable('veg_ec_layer_weights'),
                                              'Encoding/veg_ec_layer_bias':tf.get_variable('veg_ec_layer_bias'),
                                              'Encoding/sky_ec_layer_weights':tf.get_variable('sky_ec_layer_weights'),
                                              'Encoding/sky_ec_layer_bias':tf.get_variable('sky_ec_layer_bias'),
                                              'Encoding/sem_ec_layer_weights':tf.get_variable('sem_ec_layer_weights'),
                                              'Encoding/sem_ec_layer_bias':tf.get_variable('sem_ec_layer_bias'),
                                              'Encoding/full_ec_layer_weights':tf.get_variable('full_ec_layer_weights'),
                                              'Encoding/full_ec_layer_bias':tf.get_variable('full_ec_layer_bias')})

            with tf.variable_scope('Decoding',reuse=True):

                load_dc_MAE = tf.train.Saver({'Decoding/red_dc_layer_weights':tf.get_variable('red_dc_layer_weights'),
                                              'Decoding/red_dc_layer_bias':tf.get_variable('red_dc_layer_bias'),
                                              'Decoding/green_dc_layer_weights':tf.get_variable('green_dc_layer_weights'),
                                              'Decoding/green_dc_layer_bias':tf.get_variable('green_dc_layer_bias'),
                                              'Decoding/blue_dc_layer_weights':tf.get_variable('blue_dc_layer_weights'),
                                              'Decoding/blue_dc_layer_bias':tf.get_variable('blue_dc_layer_bias'),
                                              'Decoding/depth_dc_layer_weights':tf.get_variable('depth_dc_layer_weights'),
                                              'Decoding/depth_dc_layer_bias':tf.get_variable('depth_dc_layer_bias'),
                                              'Decoding/gnd_dc_layer_weights':tf.get_variable('gnd_dc_layer_weights'),
                                              'Decoding/gnd_dc_layer_bias':tf.get_variable('gnd_dc_layer_bias'),
                                              'Decoding/obj_dc_layer_weights':tf.get_variable('obj_dc_layer_weights'),
                                              'Decoding/obj_dc_layer_bias':tf.get_variable('obj_dc_layer_bias'),
                                              'Decoding/bld_dc_layer_weights':tf.get_variable('bld_dc_layer_weights'),
                                              'Decoding/bld_dc_layer_bias':tf.get_variable('bld_dc_layer_bias'),
                                              'Decoding/veg_dc_layer_weights':tf.get_variable('veg_dc_layer_weights'),
                                              'Decoding/veg_dc_layer_bias':tf.get_variable('veg_dc_layer_bias'),
                                              'Decoding/sky_dc_layer_weights':tf.get_variable('sky_dc_layer_weights'),
                                              'Decoding/sky_dc_layer_bias':tf.get_variable('sky_dc_layer_bias'),
                                              'Decoding/sem_dc_layer_weights':tf.get_variable('sem_dc_layer_weights'),
                                              'Decoding/sem_dc_layer_bias':tf.get_variable('sem_dc_layer_bias'),
                                              'Decoding/full_dc_layer_weights':tf.get_variable('full_dc_layer_weights'),
                                              'Decoding/full_dc_layer_bias':tf.get_variable('full_dc_layer_bias')})


        saver = tf.train.Saver()
        load_weights = tf.train.Saver()

        rmse_min = np.infty
        rel_min = np.infty
        no_update_count = 0

        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        
        with tf.Session(config=config) as sess:

            self.flag_is_running = True

            train_writer1 = tf.summary.FileWriter(self.logs_dir,sess.graph)
            sess.run(tf.global_variables_initializer())

            if self.load_previous == False:
                if self.model == 'old':
                    load_ec_MAE.restore(sess,'models/full/FullMAE/fullmodel.ckpt')
                    load_dc_MAE.restore(sess,'models/full/FullMAE/fullmodel.ckpt')
                else:
                    load_ec_MAE.restore(sess,'models/full/FullMAE1/fullmodel.ckpt')
                    load_dc_MAE.restore(sess,'models/full/FullMAE1/fullmodel.ckpt')


            if self.load_previous == True:
                load_ec_MAE.restore(sess,'models/rnn/previous/rnn_model.ckpt')
                load_dc_MAE.restore(sess,'models/rnn/previous/rnn_model.ckpt')


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
                             self.init_states:np.ones((1,self.state_size)),
                             normalization:norm}

                cost_dict = {self.c_w1:c_w1,
                             self.c_w2:c_w2,
                             self.c_w3:c_w3,
                             self.c_w4:c_w4,
                             self.c_w5:c_w5}

                feed_dict.update(cost_dict)

                im_pred,c_val = sess.run([output,loss_val_update],feed_dict=feed_dict)

                depth_pred = BR.invert_depth(im_pred[3],option='est')
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

                sum_lr = sess.run(summary_lr)
                train_writer1.add_summary(sum_lr,epoch)

                sess.run(epoch_loss_reset)
                time1 = datetime.now()

                in_state = np.ones((self.batch_size,self.state_size))


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

                    if self.mirroring:
                        # horizontal mirroring
                        ind_batch = np.linspace(0,self.batch_size-1,self.batch_size).astype(int)
                        ind_rand_who = np.random.choice(ind_batch,int(self.batch_size/2),replace=False)

                        imr_in = BR.horizontal_mirroring(deepcopy(imr_in),ind_rand_who,option='RNN')
                        imr_batch_label = BR.horizontal_mirroring(deepcopy(imr_batch_label),ind_rand_who,option='RNN')
                        img_in = BR.horizontal_mirroring(deepcopy(img_in),ind_rand_who,option='RNN')
                        img_batch_label = BR.horizontal_mirroring(deepcopy(img_batch_label),ind_rand_who,option='RNN')
                        imb_in = BR.horizontal_mirroring(deepcopy(imb_in),ind_rand_who,option='RNN')
                        imb_batch_label = BR.horizontal_mirroring(deepcopy(imb_batch_label),ind_rand_who,option='RNN')

                        depth_in = BR.horizontal_mirroring(deepcopy(depth_in),ind_rand_who,option='RNN')
                        depth_batch_label = BR.horizontal_mirroring(deepcopy(depth_batch_label),ind_rand_who,option='RNN')
                        depth_mask_batch = BR.horizontal_mirroring(deepcopy(depth_mask_batch),ind_rand_who,option='RNN')

                        gnd_in = BR.horizontal_mirroring(deepcopy(gnd_in),ind_rand_who,option='RNN')
                        gnd_batch_label = BR.horizontal_mirroring(deepcopy(gnd_batch_label),ind_rand_who,option='RNN')
                        obj_in = BR.horizontal_mirroring(deepcopy(obj_in),ind_rand_who,option='RNN')
                        obj_batch_label = BR.horizontal_mirroring(deepcopy(obj_batch_label),ind_rand_who,option='RNN')
                        bld_in = BR.horizontal_mirroring(deepcopy(bld_in),ind_rand_who,option='RNN')
                        bld_batch_label = BR.horizontal_mirroring(deepcopy(bld_batch_label),ind_rand_who,option='RNN')
                        veg_in = BR.horizontal_mirroring(deepcopy(veg_in),ind_rand_who,option='RNN')
                        veg_batch_label = BR.horizontal_mirroring(deepcopy(veg_batch_label),ind_rand_who,option='RNN')
                        sky_in = BR.horizontal_mirroring(deepcopy(sky_in),ind_rand_who,option='RNN')
                        sky_batch_label = BR.horizontal_mirroring(deepcopy(sky_batch_label),ind_rand_who,option='RNN')




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
                                 self.c_w5:c_w5,
                                 self.c_r:c_wr,
                                 self.c_g:c_wg,
                                 self.c_b:c_wb}

                    feed_dict.update(cost_dict)

                    # training operation (first only full encoding is trained, then (after 10 epochs) everything is trained
                    if epoch < 10:
                        _ , c, l  = sess.run([train_op0, cost, epoch_loss_update], feed_dict=feed_dict)

                    if epoch >= 10 and epoch < 20:
                        _ , c, l  = sess.run([train_op1, cost, epoch_loss_update], feed_dict=feed_dict)

                    if epoch >= 20 and epoch < 40:
                        _ , c, l = sess.run([train_op2, cost, epoch_loss_update], feed_dict=feed_dict)

                    else:
                        _ , c, l = sess.run([train_op3, cost, epoch_loss_update], feed_dict=feed_dict)

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
                                 self.init_states:np.ones((1,self.state_size)),
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
                                                                                              sky_loss_update],
                                                                                             feed_dict=feed_dict)

                    depth_pred = BR.invert_depth(im_pred[3],option='est')
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
                    imr,img,imb,g,o,b,v,s = sess.run([imr_loss.value(),img_loss.value(),imb_loss.value(),
                                                      gnd_loss.value(),obj_loss.value(),bld_loss.value(),
                                                      veg_loss.value(),sky_loss.value()])
                    val_losses = [imr,img,imb,g,o,b,v,s]
                    self.overfitting_detection(val_losses)

                    if self.imr_dw:
                        c_wr = 0.1
                    if self.img_dw:
                        c_wg = 0.1
                    if self.imb_dw:
                        c_wb = 0.1
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



                if epoch%5 == 0 and epoch > 40:

                    if error_rms < rmse_min:

                        no_update_count = 0
                        rmse_min = error_rms
                        self.specifications['Validation RMSE'] = error_rms

                        saver.save(sess,self.model_dir+'/rnn_model.ckpt')
                        saver.save(sess,'models/rnn/previous/rnn_model.ckpt')
                        json.dump(self.specifications, open(self.logs_dir+"/specs.txt",'w'))

                    elif error_rel < rel_min:

                        no_update_count = 0
                        rel_min = error_rel
                        self.specifications['number of epochs'] = epoch
                        self.specifications['Validation Rel Error'] = error_rel

                        saver.save(sess,self.model_dir+'/rnn_model.ckpt')
                        saver.save(sess,'models/rnn/previous/rnn_model.ckpt')
                        json.dump(self.specifications, open(self.logs_dir+"/specs.txt",'w'))

                    else:
                        no_update_count += 1
                        if no_update_count == 40:

                            sess.close()
                            tf.reset_default_graph()
                            break

            saver.save(sess,self.model_dir+'/rnn_model.ckpt')
            saver.save(sess,'models/rnn/previous/rnn_model.ckpt')

        sess.close()
        tf.reset_default_graph()

    def evaluate(self,data_test,model_dir,option=None):

        n_evaluations = len(data_test)

        if option == None:
            raise ValueError('no evaluation option given')

        print('==================================================')
        print('Option:', option)

        self.data_test = data_test
        self.prepare_test_data()

        input  = [self.imr_input,self.img_input,self.imb_input,self.depth_input,
                  self.gnd_input,self.obj_input,self.bld_input,self.veg_input,self.sky_input]

        output = self.network(input)

        load_weights = tf.train.Saver()

        dir = 'models/' + model_dir

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            load_weights.restore(sess,dir+'/rnn_model.ckpt') # runs from 06112017 it ist fullmodel_rnn

            print('Size of Test Set:',n_evaluations)

            error_rms = 0
            error_rel = 0

            in_state =  np.zeros((1,self.state_size))

            zeroing = np.zeros((1,self.n_rnn_steps,self.size_input))

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

                if option == 'rgb':
                    # taking only rgb as input
                    depth_in = zeroing
                    gnd_in = zeroing
                    obj_in = zeroing
                    bld_in = zeroing
                    veg_in = zeroing
                    sky_in = zeroing

                if option == 'rgbs':
                    depth_in = zeroing

                if option == 'rgbd':
                    gnd_in = zeroing
                    obj_in = zeroing
                    bld_in = zeroing
                    veg_in = zeroing
                    sky_in = zeroing

                if option == 'rgbsd':
                    pass


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
            print('==================================================')

        sess.close()
        tf.reset_default_graph()

    def evaluate_per_frame(self,data_test,model_dir,option):

        n_evaluations = len(data_test)

        if option == None:
            raise ValueError('no evaluation option given')

        print('==================================================')
        print('Option:', option)

        self.data_test = data_test
        self.prepare_test_data()

        input  = [self.imr_input,self.img_input,self.imb_input,self.depth_input,
                  self.gnd_input,self.obj_input,self.bld_input,self.veg_input,self.sky_input]

        output = self.network(input)

        load_weights = tf.train.Saver()

        dir = 'models/' + model_dir

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            load_weights.restore(sess,dir+'/rnn_model.ckpt') # runs from 06112017 it ist fullmodel_rnn

            in_state =  np.zeros((1,self.state_size))

            zeroing = np.zeros((1,self.n_rnn_steps,self.size_input))

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

                if option == 'rgb':
                    # taking only rgb as input
                    depth_in = zeroing
                    gnd_in = zeroing
                    obj_in = zeroing
                    bld_in = zeroing
                    veg_in = zeroing
                    sky_in = zeroing

                if option == 'rgbs':
                    depth_in = zeroing

                if option == 'rgbd':
                    gnd_in = zeroing
                    obj_in = zeroing
                    bld_in = zeroing
                    veg_in = zeroing
                    sky_in = zeroing

                if option == 'rgbsd':
                    pass


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

                error_rms = eval.rms_error(est,gt)
                error_rel = eval.relative_error(est,gt)

        sess.close()
        tf.reset_default_graph()

        return error_rms,error_rel


    def evaluate_sequence(self,sequence,n_rnn_steps=None,option=None,frequency=None,run=None):

        if run == None:
            raise ValueError('no run ID given')

        if option == None:
            raise ValueError('no distortion option given')

        if frequency == None:
            raise ValueError('no distortion frequency given')

        n_sets = len(sequence[0])

        label_data = sequence
        input_data = distort_test_sequences(deepcopy(label_data),n_rnn_steps=n_rnn_steps,option=option,frequency=frequency)

        # network call
        input  = [self.imr_input,self.img_input,self.imb_input,self.depth_input,
                  self.gnd_input,self.obj_input,self.bld_input,self.veg_input,self.sky_input]
        output = self.network(input)

        # preparation of load model
        load_weights = tf.train.Saver()

        dir = 'models/rnn/trained-models/' + run

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            load_weights.restore(sess,dir+'/rnn_model.ckpt') # runs from 06112017 it ist fullmodel_rnn

            error_rms = []
            error_rel = []

            in_state =  np.zeros((1,self.state_size))

            for i in range(0,n_sets):

                depth_label = label_data[3][i]
                depth_label = depth_label[n_rnn_steps-1]

                imr_in,img_in,imb_in,depth_in,gnd_in,obj_in,bld_in,veg_in,sky_in = input_distortion(copy(input_data[0][i]),
                                                                                                    copy(input_data[1][i]),
                                                                                                    copy(input_data[2][i]),
                                                                                                    copy(input_data[3][i]),
                                                                                                    copy(input_data[4][i]),
                                                                                                    copy(input_data[5][i]),
                                                                                                    copy(input_data[6][i]),
                                                                                                    copy(input_data[7][i]),
                                                                                                    copy(input_data[8][i]),
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

                inv_depth_pred = np.asarray(copy(depth_pred))
                inv_depth_label = np.asarray(copy(depth_label))

                gt = BR.invert_depth(inv_depth_label)
                est = BR.invert_depth(inv_depth_pred)

                error_rms.append(eval.rms_error(est,gt))
                error_rel.append(eval.relative_error(est,gt))


        sess.close()
        tf.reset_default_graph()

        return error_rms,error_rel




