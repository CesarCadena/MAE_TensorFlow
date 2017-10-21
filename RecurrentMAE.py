# model based on 'Multi-modal Auto-Encoders as Joint Estimators for Robotics Scene Understanding' by Cadena et al.
# code developed by Silvan Weder


import tensorflow as tf
import numpy as np
import os
import evaluation_functions as eval

from load_data import load_data
from visualization import display_frame,plot_training_loss
from input_distortion import input_distortion, pretraining_input_distortion
from datetime import datetime
from copy import copy

# LOAD DATA

data_train, data_validate, data_test = load_data()



class MAE:

    def __init__(self,data_train,data_validate,data_test,resolution=(18,60)):

        self.data_train = data_train
        self.data_validate = data_validate
        self.data_test = data_test

        self.height = resolution[0]
        self.width = resolution[1]

        self.size_input = self.height*self.width
        self.size_coding = 1024

        self.n_training_data = 300 # max 15301
        self.n_training_validations = 50

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

         # recurrent options

        self.n_rnn_steps = 2
        self.n_states = 3

        # prepare data
        self.prepare_training_data()
        self.prepare_validation_data()
        self.prepare_test_data()

        # flags
        self.flag_is_running = False




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

        # training options
        self.batch_size = 60
        self.n_batches = int(len(self.imr_train)/self.batch_size)

        # rnn initial states
        self.init_states = tf.placeholder('float',[None,self.size_coding])

        self.learning_rate = 1e-6
        self.hm_epochs = 200

        # validation options
        self.n_validation_steps = 1

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

        tf.app.flags.DEFINE_string('logs_dir',self.logs_dir,'where to store the logs')
        tf.app.flags.DEFINE_string('model_dir',self.model_dir,'where to store the trained model')

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


        self.imr_train_label = []
        self.img_train_label = []
        self.imb_train_label = []
        self.depth_train_label = []
        self.gnd_train_label = []
        self.obj_train_label = []
        self.bld_train_label = []
        self.veg_train_label = []
        self.sky_train_label = []


        for i in self.data_train:
            for j in range(1,len(i)):

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
        self.gnd_test = []
        self.obj_test = []
        self.bld_test = []
        self.veg_test = []
        self.sky_test = []


        for i in self.data_test:
            for j in i:
                self.imr_test.append(j['xcr1']/255.)
                self.img_test.append(j['xcg1']/255.)
                self.imb_test.append(j['xcb1']/255.)
                self.depth_test.append(j['xid1'])
                self.gnd_test.append((j['sem1']==1).astype(int))
                self.obj_test.append((j['sem1']==2).astype(int))
                self.bld_test.append((j['sem1']==3).astype(int))
                self.veg_test.append((j['sem1']==4).astype(int))
                self.sky_test.append((j['sem1']==5).astype(int))

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

    def encoding_network(self, imr, img, imb, depth, gnd, obj, bld, veg, sky, mode='training'):

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
                              'bias':tf.Variable(tf.zeros([self.size_input]),name='blue_dc_layer_weights')}
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



        inputs = tf.split(inputs,self.n_rnn_steps,axis=1)


        # output containers

        imr_out = []
        img_out = []
        imb_out = []
        depth_out = []
        gnd_out = []
        obj_out = []
        bld_out = []
        veg_out = []
        sky_out = []

        # forward pass

        for i in inputs:

            input = tf.reshape(i,[self.batch_size,self.size_coding])

            # full decoding neurons

            self.full_decoding = tf.add(tf.matmul(input,self.full_dc_layer['weights']),
                                        self.full_dc_layer['bias'])
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

            imr_out.append(self.red_output)
            img_out.append(self.green_output)
            imb_out.append(self.blue_output)
            depth_out.append(self.depth_output)

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

            gnd_out.append(self.gnd_output)
            obj_out.append(self.obj_output)
            bld_out.append(self.bld_output)
            veg_out.append(self.veg_output)
            sky_out.append(self.sky_output)

        output = [imr_out,img_out,imb_out,depth_out,gnd_out,obj_out,bld_out,veg_out,sky_out]
        return output

    def rnn_network(self,inputs):


        inputs = tf.stack([inputs[0],inputs[1]])

        cell = tf.nn.rnn_cell.BasicRNNCell(num_units=self.size_coding)
        states_series, current_state = tf.nn.dynamic_rnn(cell, inputs,time_major=True, initial_state=self.init_states)

        return states_series, current_state

    def collect_variables(self):

        for i in self.layers:
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, i['weights'])
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, i['bias'])

    def train_model(self):

        encoding = self.encoding_network(self.imr_input,
                                         self.img_input,
                                         self.imb_input,
                                         self.depth_input,
                                         self.gnd_input,
                                         self.obj_input,
                                         self.bld_input,
                                         self.veg_input,
                                         self.sky_input,
                                         mode='training')

        states, _current_state = self.rnn_network(encoding)
        output = self.decoding_network(states)

        imr_label_series = tf.unstack(self.imr_label,axis=1)
        img_label_series = tf.unstack(self.img_label,axis=1)
        imb_label_series = tf.unstack(self.imb_label,axis=1)
        depth_label_series = tf.unstack(self.depth_label,axis=1)
        gnd_label_series = tf.unstack(self.gnd_label,axis=1)
        obj_label_series = tf.unstack(self.obj_label,axis=1)
        bld_label_series = tf.unstack(self.bld_label,axis=1)
        veg_label_series = tf.unstack(self.veg_label,axis=1)
        sky_label_series = tf.unstack(self.sky_label,axis=1)


        cost = tf.nn.l2_loss(imr_label_series[-1]-output[0][-1]) + \
               tf.nn.l2_loss(img_label_series[-1]-output[1][-1]) + \
               tf.nn.l2_loss(imb_label_series[-1]-output[2][-1]) + \
               10*tf.nn.l2_loss(depth_label_series[-1]-output[3][-1]) + \
               tf.nn.l2_loss(gnd_label_series[-1]-output[4][-1]) + \
               tf.nn.l2_loss(obj_label_series[-1]-output[5][-1]) + \
               tf.nn.l2_loss(bld_label_series[-1]-output[6][-1]) + \
               tf.nn.l2_loss(veg_label_series[-1]-output[7][-1]) + \
               tf.nn.l2_loss(sky_label_series[-1]-output[8][-1])


        loss = tf.nn.l2_loss(imr_label_series[-1]-output[0][-1]) + \
               tf.nn.l2_loss(img_label_series[-1]-output[1][-1]) + \
               tf.nn.l2_loss(imb_label_series[-1]-output[2][-1]) + \
               10*tf.nn.l2_loss(depth_label_series[-1]-output[3][-1]) + \
               tf.nn.l2_loss(gnd_label_series[-1]-output[4][-1]) + \
               tf.nn.l2_loss(obj_label_series[-1]-output[5][-1]) + \
               tf.nn.l2_loss(bld_label_series[-1]-output[6][-1]) + \
               tf.nn.l2_loss(veg_label_series[-1]-output[7][-1]) + \
               tf.nn.l2_loss(sky_label_series[-1]-output[8][-1])


        self.collect_variables()

        regularizer = tf.contrib.layers.l2_regularizer(scale=0.0005)
        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)

        cost += reg_term
        loss += reg_term

        epoch_loss = tf.Variable(0.0,name='epoch_loss',trainable=False)
        val_loss = tf.Variable(0.0,name='val_loss',trainable=False)

        epoch_loss_reset = epoch_loss.assign(0)
        epoch_loss_update = epoch_loss.assign_add(cost)

        normalization = tf.placeholder('float')

        loss_val_reset = val_loss.assign(0)
        loss_val_update = val_loss.assign_add(loss/normalization)

        sum_epoch_loss = tf.summary.scalar('Epoch Loss Full Model',epoch_loss)
        sum_val_loss = tf.summary.scalar('Validation Loss Full Model',val_loss)

        optimizer1 = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost,var_list=[self.full_ec_layer['weights'],
                                                                                                      self.full_ec_layer['bias'],
                                                                                                      self.full_dc_layer['weights'],
                                                                                                      self.full_dc_layer['bias']])

        optimizer2 = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)

        validations = np.arange(0, self.n_training_validations)
        set_val = np.random.choice(validations,self.n_training_validations,replace=False)


        load_red = tf.train.Saver({'red_ec_layer_weights':self.red_ec_layer['weights'],
                                   'red_ec_layer_bias':self.red_ec_layer['bias'],
                                   'red_dc_layer_weights':self.red_dc_layer['weights'],
                                   'red_dc_layer_bias':self.red_dc_layer['bias']})

        load_green = tf.train.Saver({'green_ec_layer_weights':self.green_ec_layer['weights'],
                                     'green_ec_layer_bias':self.green_ec_layer['bias'],
                                     'green_dc_layer_weights':self.green_dc_layer['weights'],
                                     'green_dc_layer_bias':self.green_dc_layer['bias']})

        load_blue = tf.train.Saver({'blue_ec_layer_weights':self.blue_ec_layer['weights'],
                                    'blue_ec_layer_bias':self.blue_ec_layer['bias'],
                                    'blue_dc_layer_weights':self.blue_dc_layer['weights'],
                                    'blue_dc_layer_bias':self.blue_dc_layer['bias'],})

        load_depth = tf.train.Saver({'depth_ec_layer_weights':self.depth_ec_layer['weights'],
                                     'depth_ec_layer_bias':self.depth_ec_layer['bias'],
                                     'depth_dc_layer_weights':self.depth_dc_layer['weights'],
                                     'depth_dc_layer_bias':self.depth_dc_layer['bias']})


        load_shared_sem = tf.train.Saver({'gnd_ec_layer_weights':self.gnd_ec_layer['weights'],
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
                                            'sem_dc_layer_bias':self.sem_dc_layer['bias']})

        saver = tf.train.Saver()


        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        
        with tf.Session(config=config) as sess:

            self.flag_is_running = True

            train_writer1 = tf.summary.FileWriter(self.FLAGS.logs_dir,sess.graph)
            sess.run(tf.global_variables_initializer())

            load_red.restore(sess,'models/pretraining/pretrained_models/pretrained_red.ckpt')
            load_green.restore(sess,'models/pretraining/pretrained_models/pretrained_green.ckpt')
            load_blue.restore(sess,'models/pretraining/pretrained_models/pretrained_blue.ckpt')
            load_depth.restore(sess,'models/pretraining/pretrained_models/pretrained_depth.ckpt')
            load_shared_sem.restore(sess,'models/pretraining/pretrained_models/pretrained_shared_semantics.ckpt')

            tf.get_default_graph().finalize()

            for epoch in range(0,self.hm_epochs):
                sess.run(epoch_loss_reset)
                time1 = datetime.now()

                in_state = np.zeros((self.batch_size,self.size_coding))

                self.batch_size = 60 # adjust batch size for training


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

                    imr_batch_label = self.imr_train_label[_*self.batch_size:(_+1)*self.batch_size]
                    img_batch_label = self.img_train_label[_*self.batch_size:(_+1)*self.batch_size]
                    imb_batch_label = self.imb_train_label[_*self.batch_size:(_+1)*self.batch_size]
                    depth_batch_label = self.depth_train_label[_*self.batch_size:(_+1)*self.batch_size]
                    gnd_batch_label= self.gnd_train_label[_*self.batch_size:(_+1)*self.batch_size]
                    obj_batch_label = self.obj_train_label[_*self.batch_size:(_+1)*self.batch_size]
                    bld_batch_label = self.bld_train_label[_*self.batch_size:(_+1)*self.batch_size]
                    veg_batch_label = self.veg_train_label[_*self.batch_size:(_+1)*self.batch_size]
                    sky_batch_label = self.sky_train_label[_*self.batch_size:(_+1)*self.batch_size]

                    current_state = np.zeros((self.batch_size,self.size_coding))

                    imr_in,img_in,imb_in,depth_in,gnd_in,obj_in,bld_in,veg_in,sky_in = input_distortion(copy(imr_batch),
                                                                                                        copy(img_batch),
                                                                                                        copy(imb_batch),
                                                                                                        copy(depth_batch),
                                                                                                        copy(gnd_batch),
                                                                                                        copy(obj_batch),
                                                                                                        copy(bld_batch),
                                                                                                        copy(veg_batch),
                                                                                                        copy(sky_batch),
                                                                                                        resolution=(18,60),
                                                                                                        rnn=True)

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

                    # training operation (first only full encoding is trained, then (after 10 epochs) everything is trained
                    if epoch < 10:
                        _ , c, l, in_state = sess.run([optimizer1, cost, epoch_loss_update,_current_state], feed_dict=feed_dict)
                    else:
                        _ ,c, l, in_state = sess.run([optimizer2, cost, epoch_loss_update,_current_state],feed_dict=feed_dict)



                sum_train = sess.run(sum_epoch_loss)
                train_writer1.add_summary(sum_train,epoch)

                print('----------------------------------------------------------------')
                print('Epoch', epoch, 'completed out of', self.hm_epochs)
                print('Training Loss (per epoch): ', sess.run(epoch_loss.value()))
                time2 = datetime.now()
                delta = time2-time1
                print('Epoch Time [seconds]:', delta.seconds)
                print('-----------------------------------------------------------------')


                sess.run(loss_val_reset)

                norm = 2*8*1080*set_val.shape[0]

                self.batch_size = 1 # adjust batchsize for validation

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
                                 self.init_states:np.zeros((1,self.size_coding)),
                                 normalization:norm}

                    im_pred,c_val = sess.run([output,loss_val_update],feed_dict=feed_dict)

                sum_val = sess.run(sum_val_loss)
                train_writer1.add_summary(sum_val,epoch)
                print('Validation Loss (per pixel): ', sess.run(val_loss.value())/normalization)
                
                print('-----------------------------------------------------------------')


            if self.saving == True:
                saver.save(sess,self.FLAGS.model_dir+'/fullmodel.ckpt')
                print('SAVED MODEL')

    def validate_model(self,n_validations,run,loadmodel=True):

        with tf.Session() as sess:

            prediction = self.encoding_network(self.imr_input,
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

    def evaluate(self,run=False):

        predictions = self.encoding_network(self.imr_input,
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

            n_evaluations = 697
            print('Size of Test Set:',n_evaluations)

            error_rms = 0



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

                pred = sess.run(predictions,feed_dict=feed_dict)
                depth_pred = pred[4]


                depth_pred = np.asarray(depth_pred)
                depth_label = np.asarray(depth_label)


                error_rms += eval.rms_error(depth_pred,depth_label)

            print('Error (RMS):', error_rms/n_evaluations)









# running model

mae = MAE(data_train,data_validate,data_test)
mae.train_model()

#mae.evaluate(run='20171011-224144')





















