# model based on 'Multi-modal Auto-Encoders as Joint Estimators for Robotics Scene Understanding' by Cadena et al.
# code developed by Silvan Weder


import tensorflow as tf
import numpy as np


from load_data import load_data
from visualization import display_frame,plot_training_loss

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

        self.n_training_data = 1000 # max 15301


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

        # prepare data
        self.prepare_training_data()

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



        # training options
        self.batch_size = 60
        self.n_batches = int(len(self.imr_train)/self.batch_size)

        self.learning_rate = 1e-06
        self.n_training_epochs = 100

        # model saving
        self.saving = True
        self.folder_model = 'models'

        self.project_dir ='/Users/silvanadrianweder/Polybox/Master/02-semester/01-semester-project/02-code/mae_tensorflow'
        self.model_dir = self.project_dir + '/models'

        tf.app.flags.DEFINE_string('train_dir',self.model_dir,'where to store the trained model')
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

        t_iterator = 0

        for i in self.data_train:
            for j in i:

                if t_iterator == self.n_training_data:
                    #show_frame = display_frame(j,(self.height,self.width))
                    break
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

                t_iterator += 1

        # randomly shuffle input frames
        rand_indices = np.arange(self.n_training_data).astype(int)
        np.random.shuffle(rand_indices)

        self.imr_train = np.asarray(self.imr_train)[rand_indices].tolist()
        self.img_train = np.asarray(self.img_train)[rand_indices].tolist()
        self.imb_train = np.asarray(self.imb_train)[rand_indices].tolist()
        self.depth_train = np.asarray(self.depth_train)[rand_indices].tolist()
        self.gnd_train = np.asarray(self.gnd_train)[rand_indices].tolist()
        self.obj_train = np.asarray(self.obj_train)[rand_indices].tolist()
        self.bld_train = np.asarray(self.bld_train)[rand_indices].tolist()
        self.veg_train = np.asarray(self.veg_train)[rand_indices].tolist()
        self.sky_train = np.asarray(self.sky_train)[rand_indices].tolist()

    def true_function_1(self):

                imr_input = tf.nn.dropout(self.imr_input, (1 - self.imr_noise))
                img_input = tf.nn.dropout(self.img_input, (1 - self.imr_noise))
                imb_input = tf.nn.dropout(self.imb_input, (1 - self.imr_noise))
                depth_input = tf.nn.dropout(self.depth_input, (1 - self.imr_noise))
                gnd_input = tf.nn.dropout(self.gnd_input, (1 - self.imr_noise))
                obj_input = tf.nn.dropout(self.obj_input, (1 - self.imr_noise))
                bld_input = tf.nn.dropout(self.bld_input, (1 - self.imr_noise))
                veg_input = tf.nn.dropout(self.veg_input, (1 - self.imr_noise))
                sky_input = tf.nn.dropout(self.sky_input, (1 - self.imr_noise))
                return imr_input,img_input,imb_input,depth_input,gnd_input,obj_input,bld_input,veg_input,sky_input

    def true_function_2(self):
                shape_input = tf.shape(self.depth_input)

                imr_input = self.imr_input
                img_input = self.img_input
                imb_input = self.imb_input
                depth_input = tf.zeros(shape_input,dtype='float')
                gnd_input = tf.zeros(shape_input,dtype='float')
                obj_input = tf.zeros(shape_input,dtype='float')
                bld_input = tf.zeros(shape_input,dtype='float')
                veg_input = tf.zeros(shape_input,dtype='float')
                sky_input = tf.zeros(shape_input,dtype='float')
                return imr_input,img_input,imb_input,depth_input,gnd_input,obj_input,bld_input,veg_input,sky_input

    def false_function(self):
                imr_input = self.imr_input
                img_input = self.img_input
                imb_input = self.imb_input
                depth_input = self.depth_input
                gnd_input = self.gnd_input
                obj_input = self.obj_input
                bld_input = self.bld_input
                veg_input = self.veg_input
                sky_input = self.sky_input
                return imr_input,img_input,imb_input,depth_input,gnd_input,obj_input,bld_input,veg_input,sky_input

    def input_distortion(self):

        if self.mode == 'denoising':
            u = tf.random_uniform([],0,1)
            tf.Print(u,[u])

            border1 = tf.constant(0.2)
            border2 = tf.constant(0.5)

            imr_input,img_input,imb_input,depth_input,gnd_input,obj_input,bld_input,veg_input,sky_input = tf.cond(tf.less(u,border1),
                                                                                                                  lambda:self.true_function_1(),
                                                                                                                  lambda:tf.cond(tf.logical_and(tf.greater_equal(u,border1),tf.less(u,border2)),
                                                                                                                                   lambda:self.true_function_2(),
                                                                                                                                   lambda:self.false_function()))
        else:
            imr_input = self.imr_input
            img_input = self.img_input
            imb_input = self.imb_input
            depth_input = self.depth_input
            gnd_input = self.gnd_input
            obj_input = self.obj_input
            bld_input = self.bld_input
            veg_input = self.veg_input
            sky_input = self.sky_input

        return imr_input,img_input,imb_input,depth_input,gnd_input,obj_input,bld_input,veg_input,sky_input


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

        if mode == 'training':
            imr,img,imb,depth,gnd,obj,bld,veg,sky = self.input_distortion()



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
        self.sem_ec_layer = {'weights':tf.Variable(tf.random_normal([5*self.size_coding,self.size_coding],stddev=0.01),name='sem_ec_layer_weights'),
                             'bias' : tf.Variable(tf.zeros([self.size_coding]),name='sem_ec_layer_bias')}
        self.layers.append(self.sem_ec_layer)

        # semantics neuron (relu activation)
        self.sem_encoding = tf.add(tf.matmul(self.sem_concat,self.sem_ec_layer['weights']),
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
                              'bias':tf.Variable(tf.zeros([self.size_input]),name='blue_dc_layer_weights')}
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

        self.full_sem_dc_layer = {'weights':tf.Variable(tf.random_normal([self.size_coding,5*self.size_coding],stddev=0.01),name='full_sem_dc_layer_weights'),
                                  'bias':tf.Variable(tf.zeros([5*self.size_coding]),name='full_sem_dc_layer_bias')}
        self.layers.append(self.full_sem_dc_layer)

        # decoding neurons full semantics

        self.full_sem = tf.add(tf.matmul(self.sem_full_dc,self.full_sem_dc_layer['weights']),
                               self.full_sem_dc_layer['bias'])
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


        return self.red_output,self.green_output,self.blue_output,self.depth_output,self.gnd_output,self.obj_output,self.bld_output,self.veg_output,self.sky_output

    def collect_variables(self):

        for i in self.layers:
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, i['weights'])
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, i['bias'])

    def train_model(self):


        imr_out,img_out,imb_out,depth_out,gnd_out,obj_out,bld_out,veg_out,sky_out = self.neural_model(self.imr_input,
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

        regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)

        cost = tf.nn.l2_loss(imr_out-self.imr_input) + \
                      tf.nn.l2_loss(img_out-self.img_input) + \
                      tf.nn.l2_loss(imb_out-self.imb_input) + \
                      tf.nn.l2_loss(gnd_out-self.gnd_input) + \
                      tf.nn.l2_loss(obj_out-self.obj_input) + \
                      tf.nn.l2_loss(bld_out-self.bld_input) + \
                      tf.nn.l2_loss(veg_out-self.veg_input) + \
                      tf.nn.l2_loss(sky_out-self.sky_input) + \
                      reg_term

        cost = cost + tf.nn.l2_loss(tf.multiply(self.depth_mask,depth_out)-tf.multiply(self.depth_mask,self.depth_input)) # depth mask for loss computation

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)
        hm_epochs = self.n_training_epochs

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            epoch_losses = []
            for epoch in range(hm_epochs):
                epoch_loss = 0
                for _ in range(self.n_batches):

                    feed_dict = {self.imr_input:self.imr_train[_*self.n_batches:(_+1)*self.n_batches],
                                 self.img_input:self.img_train[_*self.n_batches:(_+1)*self.n_batches],
                                 self.imb_input:self.imb_train[_*self.n_batches:(_+1)*self.n_batches],
                                 self.depth_input:self.depth_train[_*self.n_batches:(_+1)*self.n_batches],
                                 self.gnd_input:self.gnd_train[_*self.n_batches:(_+1)*self.n_batches],
                                 self.obj_input:self.obj_train[_*self.n_batches:(_+1)*self.n_batches],
                                 self.bld_input:self.bld_train[_*self.n_batches:(_+1)*self.n_batches],
                                 self.veg_input:self.veg_train[_*self.n_batches:(_+1)*self.n_batches],
                                 self.sky_input:self.sky_train[_*self.n_batches:(_+1)*self.n_batches],
                                 self.depth_mask:self.depth_mask_train[_*self.n_batches:(_+1)*self.n_batches]}

                    _, c = sess.run([optimizer, cost], feed_dict=feed_dict)
                    epoch_loss += c

                epoch_losses.append(epoch_loss)
                print('Epoch', epoch+1, 'completed out of', hm_epochs, 'loss:', epoch_loss)



            if self.saving == True:
                plot_training_loss(epoch_losses,name='training_loss_MAE')
                saver = tf.train.Saver()
                saver.save(sess,self.FLAGS.train_dir+'/models.ckpt')
                print('SAVED MODEL')



# running model

mae = MAE(data_train,data_validate,data_test)
mae.train_model()





















