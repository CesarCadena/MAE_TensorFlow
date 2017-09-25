
# coding: utf-8

# In[16]:

import tensorflow as tf
import numpy as np
from load_data import load_data
from input_distortion import input_distortion


data_train,data_validate,data_test = load_data()


class PretrainingMAE():

    def __init__(self,data_train,data_validate,data_test):

        # storing data
        self.data_train = data_train
        self.data_val = data_validate
        self.data_test = data_test

        # training options

        self.batch_size = 128
        self.hm_epochs = 20

        self.input_size = 1080
        self.hidden_size = 1024

        self.saving = True
        self.n_training_data = 200

        self.prepare_training_data()

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


        # layers container

        self.layers = []

        # directory definitions
        self.project_dir ='.'
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
                self.gnd_train.append((j['sem2']==1).astype(int))
                self.obj_train.append((j['sem2']==2).astype(int))
                self.bld_train.append((j['sem2']==3).astype(int))
                self.veg_train.append((j['sem2']==4).astype(int))
                self.sky_train.append((j['sem2']==5).astype(int))

                t_iterator += 1

        # randomly shuffle input frames
        rand_indices = np.arange(len(self.imr_train)).astype(int)
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
        self.depth_mask_train = np.asarray(self.depth_mask_train)[rand_indices].tolist()


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

        output = tf.nn.relu(tf.add(tf.matmul(hidden_depth, self.depth_dc_layer['weights']),self.depth_dc_layer['bias']))

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

                    imr_in,img_in,imb_in,depth_in,gnd_in,obj_in,bld_in,veg_in,sky_in = input_distortion(imr_batch,
                                                                                                        img_batch,
                                                                                                        imb_batch,
                                                                                                        depth_batch,
                                                                                                        gnd_batch,
                                                                                                        obj_batch,
                                                                                                        bld_batch,
                                                                                                        veg_batch,
                                                                                                        sky_batch,
                                                                                                        border1=0.8,
                                                                                                        border2=1,
                                                                                                        resolution=(18,60))

                    feed_dict_red = {self.input_red:imr_in,
                                     self.label_red:imr_batch}

                    feed_dict_green = {self.input_green:img_in,
                                     self.label_green:img_batch}

                    feed_dict_blue = {self.input_blue:imb_in,
                                     self.label_blue:imb_batch}

                    feed_dict_depth = {self.input_depth:depth_in,
                                       self.depth_mask:self.depth_mask_train[_*self.batch_size:(_+1)*self.batch_size],
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



                    _ , c_red = sess.run([opt_red,cost_red],feed_dict=feed_dict_red)
                    epoch_loss[0] += c_red

                    _ , c_green = sess.run([opt_green,cost_green],feed_dict=feed_dict_green)
                    epoch_loss[1] += c_green

                    _ , c_blue = sess.run([opt_blue,cost_blue],feed_dict=feed_dict_blue)
                    epoch_loss[2] += c_blue

                    _ , c_depth = sess.run([opt_depth,cost_depth],feed_dict=feed_dict_depth)
                    epoch_loss[3] += c_depth

                    _ , c_gnd = sess.run([opt_gnd,cost_gnd],feed_dict=feed_dict_gnd)
                    epoch_loss[4] += c_gnd

                    _ , c_obj = sess.run([opt_obj,cost_obj],feed_dict=feed_dict_obj)
                    epoch_loss[5] += c_obj

                    _ , c_bld = sess.run([opt_bld,cost_bld],feed_dict=feed_dict_bld)
                    epoch_loss[6] += c_bld

                    _ , c_veg = sess.run([opt_veg,cost_veg],feed_dict=feed_dict_veg)
                    epoch_loss[7] += c_veg

                    _ , c_sky = sess.run([opt_sky,cost_sky],feed_dict=feed_dict_sky)
                    epoch_loss[8] += c_sky

                epoch_losses.append(epoch_loss)
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

        cost_sem = tf.nn.l2_loss(gnd_pred-self.label_gnd) + \
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
        cost_sem += reg_sem

        optimizer1 = tf.train.AdamOptimizer(learning_rate=0.000001)
        opt_sem1 = optimizer1.minimize(cost_sem,var_list=[self.sem_ec_layer['weights'],
                                                          self.sem_ec_layer['bias'],
                                                          self.sem_dc_layer['weights'],
                                                          self.sem_dc_layer['bias']])

        optimizer2 = tf.train.AdamOptimizer(learning_rate=0.000001)
        opt_sem2 = tf.train.AdamOptimizer(learning_rate=0.000001).minimize(cost_sem)

        saver_load = tf.train.Saver({'gnd_ec_layer_weights':self.gnd_ec_layer['weights'],
                                     'gnd_ec_layer_bias':self.gnd_ec_layer['bias'],
                                     'obj_ec_layer_weights':self.obj_ec_layer['weights'],
                                     'obj_ec_layer_bias':self.obj_ec_layer['bias'],
                                     'bld_ec_layer_weights':self.bld_ec_layer['weights'],
                                     'bld_ec_layer_bias':self.bld_ec_layer['bias'],
                                     'veg_ec_layer_weights':self.veg_ec_layer['weights'],
                                     'veg_ec_layer_bias':self.veg_ec_layer['bias'],
                                     'sky_ec_layer_weights':self.sky_ec_layer['weights'],
                                     'sky_ec_layer_bias':self.sky_ec_layer['bias'],
                                     'gnd_dc_layer_weights':self.gnd_dc_layer['weights'],
                                     'gnd_dc_layer_bias':self.gnd_dc_layer['bias'],
                                     'obj_dc_layer_weights':self.obj_dc_layer['weights'],
                                     'obj_dc_layer_bias':self.obj_dc_layer['bias'],
                                     'bld_dc_layer_weights':self.bld_dc_layer['weights'],
                                     'bld_dc_layer_bias':self.bld_dc_layer['bias'],
                                     'veg_dc_layer_weights':self.veg_dc_layer['weights'],
                                     'veg_dc_layer_bias':self.veg_dc_layer['bias'],
                                     'sky_dc_layer_weights':self.sky_dc_layer['weights'],
                                     'sky_dc_layer_bias':self.sky_dc_layer['bias']})

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver_load.restore(sess,self.FLAGS.train_dir+'/pretrained1.ckpt')

            n_batches = int(len(self.imr_train)/self.batch_size)
            epoch_losses = []

            for epoch in range(0,self.hm_epochs):
                epoch_loss = 0
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

                    imr_in,img_in,imb_in,depth_in,gnd_in,obj_in,bld_in,veg_in,sky_in = input_distortion(imr_batch,
                                                                                                        img_batch,
                                                                                                        imb_batch,
                                                                                                        depth_batch,
                                                                                                        gnd_batch,
                                                                                                        obj_batch,
                                                                                                        bld_batch,
                                                                                                        veg_batch,
                                                                                                        sky_batch,
                                                                                                        border1=0.8,
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
                        _, c = sess.run([opt_sem1,cost_sem],feed_dict=feed_dict_sem)
                    else:
                        _, c = sess.run([opt_sem2,cost_sem],feed_dict=feed_dict_sem)

                    epoch_loss += c

                epoch_losses.append(epoch_loss)
                print('Epoch', epoch, 'completed out of', self.hm_epochs, 'Loss: ', epoch_loss)




            if self.saving==True:
                saver_save = tf.train.Saver()
                saver_save.save(sess,self.FLAGS.train_dir+'/pretrained2.ckpt')
                print('SAVED MODEL')








pretraining = PretrainingMAE(data_train, data_validate, data_test)
pretraining.pretrain_seperate_channels()
pretraining.pretrain_shared_semantics()
