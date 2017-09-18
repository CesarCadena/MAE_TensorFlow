import tensorflow as tf
import numpy as np


from load_data import load_data

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

        self.n_training_data = 800

        # prepare data
        self.prepare_data()

        # placeholder definition
        self.x = tf.placeholder('float',[None,9*self.size_input])
        self.y = tf.placeholder('float',[None,9*self.size_input])



        self.batch_size = 60
        self.n_batches = int(len(self.training_frames)/self.batch_size)

    def prepare_data(self):

        training_frames = []

        for i in self.data_train:
            for j in i:
                imr = j['xcr1']
                img = j['xcg1']
                imb = j['xcb1']
                depth = j['xid1']
                gnd = (j['sem1']==1).astype(int)
                obj = (j['sem1']==2).astype(int)
                bld = (j['sem1']==3).astype(int)
                veg = (j['sem1']==4).astype(int)
                sky = (j['sem1']==5).astype(int)

                frame = np.hstack((imr,img,imb,depth,gnd,obj,bld,veg,sky))
                training_frames.append(frame)

        self.training_frames = training_frames[0:self.n_training_data]



    def neural_model(self,x):

        # split input vector into all different modalities
        imr,img,imb,depth,gnd,obj,bld,veg,sky = tf.split(x,9,axis=1)

        # semantics weights
        self.gnd_ec_layer = {'weights':tf.Variable(tf.random_normal([self.size_input,self.size_coding])),
                             'bias':tf.Variable(tf.random_normal([self.size_coding]))}

        self.obj_ec_layer = {'weights':tf.Variable(tf.random_normal([self.size_input,self.size_coding])),
                             'bias':tf.Variable(tf.random_normal([self.size_coding]))}

        self.bld_ec_layer = {'weights':tf.Variable(tf.random_normal([self.size_input,self.size_coding])),
                             'bias':tf.Variable(tf.random_normal([self.size_coding]))}

        self.veg_ec_layer = {'weights':tf.Variable(tf.random_normal([self.size_input,self.size_coding])),
                             'bias':tf.Variable(tf.random_normal([self.size_coding]))}

        self.sky_ec_layer = {'weights':tf.Variable(tf.random_normal([self.size_input,self.size_coding])),
                             'bias':tf.Variable(tf.random_normal([self.size_coding]))}


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
        self.sem_ec_layer = {'weights':tf.Variable(tf.random_normal([5*self.size_coding,self.size_coding])),
                             'bias' : tf.Variable(tf.random_normal([self.size_coding]))}

        # semantics neuron (relu activation)
        self.sem_encoding = tf.add(tf.matmul(self.sem_concat,self.sem_ec_layer['weights']),
                                   self.sem_ec_layer['bias'])
        self.sem_encoding = tf.nn.relu(self.sem_encoding)


        # depth and rgb encoding weights
        self.red_ec_layer = {'weights':tf.Variable(tf.random_normal([self.size_input,self.size_coding])),
                             'bias':tf.Variable(tf.random_normal([self.size_coding]))}

        self.green_ec_layer = {'weights':tf.Variable(tf.random_normal([self.size_input,self.size_coding])),
                               'bias':tf.Variable(tf.random_normal([self.size_coding]))}

        self.blue_ec_layer = {'weights':tf.Variable(tf.random_normal([self.size_input,self.size_coding])),
                              'bias':tf.Variable(tf.random_normal([self.size_coding]))}

        self.depth_ec_layer = {'weights':tf.Variable(tf.random_normal([self.size_input,self.size_coding])),
                               'bias':tf.Variable(tf.random_normal([self.size_coding]))}

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

        self.full_ec_layer = {'weights':tf.Variable(tf.random_normal([5*self.size_coding,self.size_coding])),
                              'bias' : tf.Variable(tf.random_normal([self.size_coding]))}

        # full encoding neurons

        self.full_encoding = tf.add(tf.matmul(self.full_concat,self.full_ec_layer['weights']),
                                    self.full_ec_layer['bias'])
        self.full_encoding = tf.nn.relu(self.full_encoding)

        # full decoding

        self.full_dc_layer = {'weights':tf.Variable(tf.random_normal([self.size_coding,5*self.size_coding])),
                              'bias':tf.Variable(tf.random_normal([5*self.size_coding]))}

        # full decoding neurons

        self.full_decoding = tf.add(tf.matmul(self.full_encoding,self.full_dc_layer['weights']),
                                    self.full_dc_layer['bias'])
        self.full_decoding = tf.nn.relu(self.full_decoding)

        # slicing full decoding

        self.depth_full_dc,self.red_full_dc,self.green_full_dc,self.blue_full_dc,self.sem_full_dc = tf.split(self.full_decoding,5,1)


        # decoding layers depth and rgb

        self.red_dc_layer = {'weights':tf.Variable(tf.random_normal([self.size_coding,self.size_input])),
                             'bias':tf.Variable(tf.random_normal([self.size_input]))}

        self.green_dc_layer = {'weights':tf.Variable(tf.random_normal([self.size_coding,self.size_input])),
                               'bias':tf.Variable(tf.random_normal([self.size_input]))}

        self.blue_dc_layer = {'weights':tf.Variable(tf.random_normal([self.size_coding,self.size_input])),
                              'bias':tf.Variable(tf.random_normal([self.size_input]))}

        self.depth_dc_layer = {'weights':tf.Variable(tf.random_normal([self.size_coding,self.size_input])),
                               'bias':tf.Variable(tf.random_normal([self.size_input]))}

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
        self.depth_output = tf.sigmoid(self.depth_output)


        # decoding layer full semantics

        self.full_sem_dc_layer = {'weights':tf.Variable(tf.random_normal([self.size_coding,5*self.size_coding])),
                                  'bias':tf.Variable(tf.random_normal([5*self.size_coding]))}

        # decoding neurons full semantics

        self.full_sem = tf.add(tf.matmul(self.sem_full_dc,self.full_sem_dc_layer['weights']),
                               self.full_sem_dc_layer['bias'])

        self.full_sem = tf.nn.relu(self.full_sem)

        # splitting full semantics

        self.gnd_dc, self.obj_dc, self.bld_dc, self.veg_dc, self.sky_dc = tf.split(self.full_sem,5,axis=1)

        # decoding layers semantics

        self.gnd_dc_layer = {'weights':tf.Variable(tf.random_normal([self.size_coding,self.size_input])),
                             'bias':tf.Variable(tf.random_normal([self.size_input]))}

        self.obj_dc_layer = {'weights':tf.Variable(tf.random_normal([self.size_coding,self.size_input])),
                             'bias':tf.Variable(tf.random_normal([self.size_input]))}

        self.bld_dc_layer = {'weights':tf.Variable(tf.random_normal([self.size_coding,self.size_input])),
                             'bias':tf.Variable(tf.random_normal([self.size_input]))}

        self.veg_dc_layer = {'weights':tf.Variable(tf.random_normal([self.size_coding,self.size_input])),
                             'bias':tf.Variable(tf.random_normal([self.size_input]))}

        self.sky_dc_layer = {'weights':tf.Variable(tf.random_normal([self.size_coding,self.size_input])),
                             'bias':tf.Variable(tf.random_normal([self.size_input]))}

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

        output = tf.concat([self.red_output,
                            self.green_output,
                            self.blue_output,
                            self.depth_output,
                            self.gnd_output,
                            self.obj_output,
                            self.bld_output,
                            self.veg_output,
                            self.sky_output],
                           axis=1)

        return output

    def train_model(self):

        predictions = self.neural_model(self.x)
        cost = tf.reduce_mean(tf.pow((predictions-self.y),2))

        optimizer = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(cost)
        hm_epochs = 100

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())


            for epoch in range(hm_epochs):
                epoch_loss = 0
                for _ in range(self.n_batches):
                    epoch_frames = self.training_frames[_*self.batch_size:_*self.batch_size+self.batch_size]
                    _, c = sess.run([optimizer, cost], feed_dict={self.x:epoch_frames,self.y:epoch_frames})
                    epoch_loss += c

                print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)








mae = MAE(data_train,data_validate,data_test)
mae.train_model()





















