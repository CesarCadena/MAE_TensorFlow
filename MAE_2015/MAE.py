# model based on 'Multi-modal Auto-Encoders as Joint Estimators for Robotics Scene Understanding' by Cadena et al.
# code developed by Silvan Weder


import tensorflow as tf
import numpy as np
import os
import evaluation_functions as Eval
import basic_routines as BR

#from visualization import display_frame,plot_training_loss
from input_distortion import input_distortion, pretraining_input_distortion
from basic_routines import horizontal_mirroring, zeroing_channel
from datetime import datetime
from copy import copy


class MAE:

    def __init__(self,resolution=(18,60)):

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

        tf.app.flags.DEFINE_string('logs_dir',self.logs_dir,'where to store the logs')
        tf.app.flags.DEFINE_string('model_dir',self.model_dir,'where to store the trained model')

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
                self.imr_train.append(j['xcrLeft']/255.)
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
                self.imr_val.append(j['xcrLeft']/255.)
                self.img_val.append(j['xcgLeft']/255.)
                self.imb_val.append(j['xcbLeft']/255.)
                self.depth_val.append(j['xidLeft'])
                self.depth_mask_val.append(j['xmaskLeft'])
                self.gnd_val.append((j['semLeft']==1).astype(int))
                self.obj_val.append((j['semLeft']==2).astype(int))
                self.bld_val.append((j['semLeft']==3).astype(int))
                self.veg_val.append((j['semLeft']==4).astype(int))
                self.sky_val.append((j['semLeft']==5).astype(int))
                
                self.imr_val.append(j['xcrRight']/255.)
                self.img_val.append(j['xcgRight']/255.)
                self.imb_val.append(j['xcbRight']/255.)
                self.depth_val.append(j['xidRight'])
                self.depth_mask_val.append(j['xmaskRight'])
                self.gnd_val.append((j['semRight']==1).astype(int))
                self.obj_val.append((j['semRight']==2).astype(int))
                self.bld_val.append((j['semRight']==3).astype(int))
                self.veg_val.append((j['semRight']==4).astype(int))
                self.sky_val.append((j['semRight']==5).astype(int))

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
            for j in i:
                self.imr_test.append(j['xcrLeft']/255.)
                self.img_test.append(j['xcgLeft']/255.)
                self.imb_test.append(j['xcbLeft']/255.)
                self.depth_test.append(j['xidLeft'])
                self.gnd_test.append((j['semLeft']==1).astype(int))
                self.obj_test.append((j['semLeft']==2).astype(int))
                self.bld_test.append((j['semLeft']==3).astype(int))
                self.veg_test.append((j['semLeft']==4).astype(int))
                self.sky_test.append((j['semLeft']==5).astype(int))
                
                self.imr_test.append(j['xcrRight']/255.)
                self.img_test.append(j['xcgRight']/255.)
                self.imb_test.append(j['xcbRight']/255.)
                self.depth_test.append(j['xidRight'])
                self.gnd_test.append((j['semRight']==1).astype(int))
                self.obj_test.append((j['semRight']==2).astype(int))
                self.bld_test.append((j['semRight']==3).astype(int))
                self.veg_test.append((j['semRight']==4).astype(int))
                self.sky_test.append((j['semRight']==5).astype(int))

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
        
    def neural_model(self,imr,img,imb,depth,gnd,obj,bld,veg,sky,mode='training'):

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

        # list to store all layers of the MAE neural network
        self.layers = []

        # split input vector into all different modalities
        #imr,img,imb,depth,gnd,obj,bld,veg,sky = tf.split(x,9,axis=1)

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


        # semantics neurons (relu activation)
        self.gnd_encoding = tf.add(tf.matmul(gnd,self.gnd_ec_layer['weights']),
                                   self.gnd_ec_layer['bias'])
        self.gnd_encoding = tf.nn.relu(self.gnd_encoding)

        self.obj_encoding = tf.add(tf.matmul(obj,self.obj_ec_layer['weights']),
                                   self.obj_ec_layer['bias'])
        self.obj_encoding = tf.nn.relu(self.obj_encoding)

        self.bld_encoding = tf.add(tf.matmul(bld,self.bld_ec_layer['weights']),
                                   self.bld_ec_layer['bias'])
        self.bld_encoding = tf.nn.relu(self.bld_encoding)

        self.veg_encoding = tf.add(tf.matmul(veg,self.veg_ec_layer['weights']),
                                   self.veg_ec_layer['bias'])
        self.veg_encoding = tf.nn.relu(self.veg_encoding)

        self.sky_encoding = tf.add(tf.matmul(sky,self.sky_ec_layer['weights']),
                                   self.sky_ec_layer['bias'])
        self.sky_encoding = tf.nn.relu(self.sky_encoding)

        # semantics concatenate
        self.sem_concat = tf.concat([self.gnd_encoding,
                                     self.obj_encoding,
                                     self.bld_encoding,
                                     self.veg_encoding,
                                     self.sky_encoding],
                                    axis=1)

        # semantics encoding
        self.sem_ec_layer = {'weights':tf.Variable(tf.random_normal([5 * self.size_coding, self.size_coding], stddev=0.01), name='sem_ec_layer_weights'),
                             'bias' : tf.Variable(tf.zeros([self.size_coding]),name='sem_ec_layer_bias')}
        self.layers.append(self.sem_ec_layer)

        # semantics neuron (relu activation)
        self.sem_encoding = tf.add(tf.matmul(self.sem_concat, self.sem_ec_layer['weights']),
                                   self.sem_ec_layer['bias'])
        self.sem_encoding = tf.nn.relu(self.sem_encoding)


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

        # depth and rgb neurons (relu activation)
        self.red_encoding = tf.add(tf.matmul(imr,self.red_ec_layer['weights']),
                                   self.red_ec_layer['bias'])
        self.red_encoding = tf.nn.relu(self.red_encoding)

        self.green_encoding = tf.add(tf.matmul(img,self.green_ec_layer['weights']),
                                     self.green_ec_layer['bias'])
        self.green_encoding = tf.nn.relu(self.green_encoding)

        self.blue_encoding = tf.add(tf.matmul(imb,self.blue_ec_layer['weights']),
                                    self.blue_ec_layer['bias'])
        self.blue_encoding = tf.nn.relu(self.blue_encoding)

        self.depth_encoding = tf.add(tf.matmul(depth,self.depth_ec_layer['weights']),
                                     self.depth_ec_layer['bias'])

        # full concatenation

        self.full_concat = tf.concat([self.depth_encoding,self.red_encoding,self.green_encoding,self.blue_encoding,self.sem_encoding],
                                     axis=1)

        # full encoding

        self.full_ec_layer = {'weights':tf.Variable(tf.random_normal([5*self.size_coding,self.size_coding],stddev=0.01),name='full_ec_layer_weights'),
                              'bias' : tf.Variable(tf.zeros([self.size_coding]),name='full_ec_layer_bias')}
        self.layers.append(self.full_ec_layer)

        # full encoding neurons

        self.full_encoding = tf.add(tf.matmul(self.full_concat,self.full_ec_layer['weights']),
                                    self.full_ec_layer['bias'])
        self.full_encoding = tf.nn.relu(self.full_encoding)

        # full decoding

        self.full_dc_layer = {'weights':tf.Variable(tf.random_normal([self.size_coding,5*self.size_coding],stddev=0.01),name='full_dc_layer_weights'),
                              'bias':tf.Variable(tf.zeros([5*self.size_coding]),name='full_dc_layer_bias')}
        self.layers.append(self.full_dc_layer)

        # full decoding neurons

        self.full_decoding = tf.add(tf.matmul(self.full_encoding,self.full_dc_layer['weights']),
                                    self.full_dc_layer['bias'])
        self.full_decoding = tf.nn.relu(self.full_decoding)

        # slicing full decoding

        self.depth_full_dc,self.red_full_dc,self.green_full_dc,self.blue_full_dc,self.sem_full_dc = tf.split(self.full_decoding,5,1)


        # decoding layers depth and rgb

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


        # decoding layer full semantics

        self.sem_dc_layer = {'weights':tf.Variable(tf.random_normal([self.size_coding, 5 * self.size_coding], stddev=0.01), name='sem_dc_layer_weights'),
                                  'bias':tf.Variable(tf.zeros([5*self.size_coding]),name='sem_dc_layer_bias')}
        self.layers.append(self.sem_dc_layer)

        # decoding neurons full semantics

        self.full_sem = tf.add(tf.matmul(self.sem_full_dc, self.sem_dc_layer['weights']),
                               self.sem_dc_layer['bias'])
        self.full_sem = tf.nn.relu(self.full_sem)

        # splitting full semantics

        self.gnd_dc, self.obj_dc, self.bld_dc, self.veg_dc, self.sky_dc = tf.split(self.full_sem,5,axis=1)

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


        return [self.red_output,self.green_output,self.blue_output,self.depth_output,self.gnd_output,self.obj_output,self.bld_output,self.veg_output,self.sky_output]

    def collect_variables(self):

        for i in self.layers:
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, i['weights'])
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, i['bias'])

    def train_model(self,data_train,data_validate,load='none',run = ''):
        
        # prepare data
        self.prepare_training_data(data_train)
        self.prepare_validation_data(data_validate)
        
        # training options
        self.batch_size = 60
        self.n_batches = int(len(self.imr_train)/self.batch_size)

        self.learning_rate = 1e-6
        self.hm_epochs = 600
        self.hm_epoch_init = 20

        # validation options        

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


        self.collect_variables()

        regularizer = tf.contrib.layers.l2_regularizer(scale=0.0005)
        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)

        cost = tf.nn.l2_loss(prediction[0]-self.imr_label) + \
                      tf.nn.l2_loss(prediction[1]-self.img_label) + \
                      tf.nn.l2_loss(prediction[2]-self.imb_label) + \
                      tf.nn.l2_loss(prediction[4]-self.gnd_label) + \
                      tf.nn.l2_loss(prediction[5]-self.obj_label) + \
                      tf.nn.l2_loss(prediction[6]-self.bld_label) + \
                      tf.nn.l2_loss(prediction[7]-self.veg_label) + \
                      tf.nn.l2_loss(prediction[8]-self.sky_label) + \
                      reg_term


        # depth mask for loss computation
        cost = cost + 10*tf.nn.l2_loss(tf.multiply(self.depth_mask,prediction[3])-tf.multiply(self.depth_mask,self.depth_label))

        loss = tf.nn.l2_loss(prediction[0]-self.imr_label) + \
                      tf.nn.l2_loss(prediction[1]-self.img_label) + \
                      tf.nn.l2_loss(prediction[2]-self.imb_label) + \
                      tf.nn.l2_loss(prediction[4]-self.gnd_label) + \
                      tf.nn.l2_loss(prediction[5]-self.obj_label) + \
                      tf.nn.l2_loss(prediction[6]-self.bld_label) + \
                      tf.nn.l2_loss(prediction[7]-self.veg_label) + \
                      tf.nn.l2_loss(prediction[8]-self.sky_label)

        loss = loss + 10*tf.nn.l2_loss(tf.multiply(self.depth_mask,prediction[3])-tf.multiply(self.depth_mask,self.depth_label))

        epoch_loss = tf.Variable(0.0,name='epoch_loss',trainable=False)
        val_loss = tf.Variable(0.0,name='val_loss',trainable=False)

        epoch_loss_reset = epoch_loss.assign(0)
        epoch_loss_update = epoch_loss.assign_add(cost)

        normalization = tf.placeholder('float')

        loss_val_reset = val_loss.assign(0)
        loss_val_update = val_loss.assign_add(loss/normalization)

        sum_epoch_loss = tf.summary.scalar('Epoch_Loss_Full_Model',epoch_loss)
        sum_val_loss = tf.summary.scalar('Validation_Loss_Full_Model',val_loss)

        optimizer0 = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)
        
        optimizer1 = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost,var_list=[self.full_ec_layer['weights'],
                                                                                                      self.full_ec_layer['bias'],
                                                                                                      self.full_dc_layer['weights'],
                                                                                                      self.full_dc_layer['bias']])

        optimizer2 = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)
        
        optimizer3 = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)
        
        ind_batch = np.arange(0,self.batch_size)

        validations = np.arange(0, len(self.imr_val))        
        set_val = np.random.choice(validations,len(self.imr_val),replace=False)
        
        if load == 'pretrained':
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
        if load == 'continue':
            load_full = tf.train.Saver({'red_ec_layer_weights':self.red_ec_layer['weights'],
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
                                        'full_ec_layer_weights':self.full_ec_layer['weights'],
                                        'full_ec_layer_bias':self.full_ec_layer['bias'],
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
                                        'sem_dc_layer_bias':self.sem_dc_layer['bias'], #typo -> should be sem_dc_layer_bias
                                        'full_dc_layer_weights':self.full_dc_layer['weights'],
                                        'full_dc_layer_bias':self.full_dc_layer['bias']})
            saver = tf.train.Saver()
        
        
        


        config = tf.ConfigProto(log_device_placement=False)
        #config.gpu_options.per_process_gpu_memory_fraction = 0.5
        with tf.Session(config=config) as sess:

            train_writer1 = tf.summary.FileWriter(self.FLAGS.logs_dir,sess.graph)
            sess.run(tf.global_variables_initializer())
            
            if load == 'pretrained':
                folder = 'models/pretraining/pretrained_models/' + run
                load_red.restore(sess, folder + 'pretrained_red.ckpt')
                load_green.restore(sess, folder + 'pretrained_green.ckpt')
                load_blue.restore(sess, folder + 'pretrained_blue.ckpt')
                load_depth.restore(sess, folder + 'pretrained_depth.ckpt')
                load_shared_sem.restore(sess, folder + 'pretrained_shared_semantics.ckpt')
            if load == 'continue':
                folder = 'models/full/' + run + '/'
                load_full.restore(sess, folder + 'fullmodel.ckpt')                

            tf.get_default_graph().finalize()

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
                    ind_rand_who = np.random.choice(ind_batch,self.batch_size/2,replace=False)
                    
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
                    
                    if epoch >= self.hm_epoch_init:
                        # zeroing depth inputs
                        ind_rand_who = np.random.choice(ind_batch,0.75*self.batch_size,replace=False)
                        
                        depth_in = zeroing_channel(depth_in,ind_rand_who)                   
                        
                        # zeroing semantic inputs
                        ind_rand_who = np.random.choice(ind_batch,0.75*self.batch_size,replace=False)
                        
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

                print('----------------------------------------------------------------')
                print('Epoch', epoch, 'completed out of', self.hm_epochs)
                print('Training Loss (per epoch): ', sess.run(epoch_loss.value()))

                sess.run(loss_val_reset)

                for i in set_val:

                    norm = 8.*1080*set_val.shape[0]

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
                                 normalization:norm}
                   
                    im_pred,c_val = sess.run([prediction,loss_val_update],feed_dict=feed_dict)

                sum_val = sess.run(sum_val_loss)
                train_writer1.add_summary(sum_val,epoch)
                print('Validation Loss (per pixel): ', sess.run(val_loss.value()))
                time2 = datetime.now()
                delta = time2-time1
                print('Epoch Time [seconds]:', delta.seconds)
                print('-----------------------------------------------------------------')


            if self.saving == True:
                saver.save(sess,self.FLAGS.model_dir+'/fullmodel.ckpt')
                print('SAVED MODEL')

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

            n_evaluations = self.imr_test.shape[0]
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

            gt = BR.invert_depth(all_inv_depth_label)
            est = BR.invert_depth(all_inv_depth_pred)

            error_rms = Eval.rms_error(est,gt)
            error_rel = Eval.relative_error(est,gt)


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

            error_rms = Eval.rms_error(depth_est,depth_gt)
            error_rel = Eval.relative_error(depth_est,depth_gt)
            
            iu_semantics, inter, union = Eval.inter_union(all_sem_pred,all_sem_label)
            
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

# running model

#mae = MAE(data_train,data_validate,data_test)
#mae.train_model()
#mae.evaluate(run='20171016-131619')





















