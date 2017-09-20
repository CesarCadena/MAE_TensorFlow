import tensorflow as tf
import numpy as np
import pykitti

import scipy.misc as msc
import matplotlib.pyplot as plt
from load_data import load_data

# LOAD DATA

data_train, data_validate, data_test = load_data()

training_size = 300
data_train = data_train[0:training_size]

class rgb_AE():

    def __init__(self, data_train, channel='red'):

        self.data_train = []
        for i in data_train:
            for j in i:
                if channel == 'red':
                    self.data_train.append(j['xcr1'])
                if channel == 'green':
                    self.data_train.append(j['xcg1'])
                if channel == 'blue':
                    self.data_train.append(j['xcb1'])

        self.size_input = 1080
        self.size_coding = 1024
        self.size_output = self.size_input

        self.x = tf.placeholder('float',[None,self.size_input])
        self.y = tf.placeholder('float')

        self.batch_size = 60
        self.n_batches = int(len(self.data_train)/self.batch_size)


    def neural_model(self,data):

        self.layers = []

        self.coding_layer = {'weights': tf.Variable(tf.random_normal([self.size_input,self.size_coding],stddev=0.01)),
                             'bias' : tf.Variable(tf.zeros([self.size_coding]))}

        self.layers.append(self.coding_layer)

        self.decoding_layer = {'weights': tf.Variable(tf.random_normal([self.size_coding,self.size_output],stddev=0.01)),
                               'bias': tf.Variable(tf.zeros([self.size_input]))}

        self.layers.append(self.decoding_layer)


        self.coding = tf.add(tf.matmul(data,self.coding_layer['weights']),self.coding_layer['bias'])
        self.coding = tf.nn.relu(self.coding)

        self.decoding = tf.add(tf.matmul(self.coding,self.decoding_layer['weights']),self.decoding_layer['bias'])
        self.decoding = tf.nn.sigmoid(self.decoding)

        return self.decoding

    def collect_variables(self):

        for i in self.layers:
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, i['weights'])
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, i['bias'])


    def train_network(self):

        prediction = self.neural_model(self.x)

        self.collect_variables()

        regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)

        cost = tf.nn.l2_loss(self.y-prediction) + reg_term

        optimizer = tf.train.AdamOptimizer().minimize(cost)
        hm_epochs = 100

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())


            for epoch in range(hm_epochs):
                epoch_loss = 0
                for i in range(self.n_batches):
                    data_batch = self.data_train[i*self.batch_size:i*self.batch_size+self.batch_size]
                    epoch_x = data_batch
                    epoch_y = data_batch

                    _, c = sess.run([optimizer, cost], feed_dict={self.x: epoch_x, self.y: epoch_y})
                    epoch_loss += c

                print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)



red_AE = rgb_AE(data_train,channel='red')
green_AE = rgb_AE(data_train,channel='green')
blue_AE = rgb_AE(data_train,channel='blue')


red_AE.train_network()
green_AE.train_network()
blue_AE.train_network()
