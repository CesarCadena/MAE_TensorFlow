import tensorflow as tf
import numpy as np
from process_data import process_data
import pandas as pd
import scipy.io
import os
import sys
tf.reset_default_graph()

batch_size=128
hidden_size=1024
num_epoch=50
FLAG='training'
print ("start loading the data ......")
if (True):
    data=process_data(FLAG)
    Red_data=data['Red']
    Green_data=data['Green']
    Blue_data=data['Blue']


print ("Finish loading the data !")
Red_input=tf.placeholder(tf.float32,shape=[None,1080])

Red_weights=tf.Variable(tf.random_normal(shape=[1080,hidden_size],
                                  stddev=0.01),name="Red_weights")
Red_bias=tf.Variable(tf.zeros([1,hidden_size]),name="Red_bias")

Red_hidden=tf.nn.relu(tf.matmul(Red_input,Red_weights)+Red_bias)

Red_outweights=tf.Variable(tf.random_normal(shape=[hidden_size,1080],
                                  stddev=0.01),name="Red_outweights")
Red_outbias=tf.Variable(tf.zeros([1,1080]),name="Red_outbias")
Red_out=tf.matmul(Red_hidden,Red_outweights)+Red_outbias

loss_red=tf.nn.l2_loss(Red_input-Red_out)
regularization_red=(tf.nn.l2_loss(Red_weights)+tf.nn.l2_loss(Red_outweights))

loss_red_t=loss_red+1e-4*regularization_red

optimizer_red=tf.train.AdamOptimizer(learning_rate=1e-4,
                                      beta1=0.9,beta2=0.999,
                                      epsilon=1e-8,use_locking=False,
                                      name='Adam').minimize(loss_red_t)
""" Green model :"""
Green_input=tf.placeholder(tf.float32,shape=[None,1080])

Green_weights=tf.Variable(tf.random_normal(shape=[1080,hidden_size],
                                  stddev=0.01),name="Green_weights")
Green_bias=tf.Variable(tf.zeros([1,hidden_size]),name="Green_bias")

Green_hidden=tf.nn.relu(tf.matmul(Green_input,Green_weights)+Green_bias)

Green_outweights=tf.Variable(tf.random_normal(shape=[hidden_size,1080],
                                  stddev=0.01),name="Green_outweights")
Green_outbias=tf.Variable(tf.zeros([1,1080]),name="Green_outbias")
Green_out=tf.matmul(Green_hidden,Green_outweights)+Green_outbias
loss_green=tf.nn.l2_loss(Green_input-Green_out)

regularization_green=tf.nn.l2_loss(Green_weights)+tf.nn.l2_loss(Green_outweights)

loss_green_t=loss_green+1e-4*regularization_green

optimizer_green=tf.train.AdamOptimizer(learning_rate=1e-4,
                                      beta1=0.9,beta2=0.999,
                                      epsilon=1e-8,use_locking=False,
                                      name='Adam').minimize(loss_green_t)

"""Blue Channel:"""

Blue_input=tf.placeholder(tf.float32,shape=[None,1080])

Blue_weights=tf.Variable(tf.random_normal(shape=[1080,hidden_size],
                                  stddev=0.01),name="Blue_weights")

Blue_bias=tf.Variable(tf.zeros([1,hidden_size]),name="Blue_bias")

Blue_hidden=tf.nn.relu(tf.matmul(Blue_input,Blue_weights)+Blue_bias)

Blue_outweights=tf.Variable(tf.random_normal(shape=[hidden_size,1080],
                                   stddev=0.01),name="Blue_outweights")
Blue_outbias=tf.Variable(tf.zeros([1,1080]),name="Blue_outbias")
Blue_out=tf.matmul(Blue_hidden,Blue_outweights)+Blue_outbias

blue_pixel_loss=tf.nn.l2_loss(Blue_input-Blue_out)/(1080*batch_size) # [batchsize,1080]
loss_blue=tf.nn.l2_loss(Blue_input-Blue_out)
regularization_blue=tf.nn.l2_loss(Blue_weights)+tf.nn.l2_loss(Blue_outweights)

loss_blue_t=loss_blue+1e-4*regularization_blue

optimizer_blue=tf.train.AdamOptimizer(learning_rate=1e-4,
                                      beta1=0.9,beta2=0.999,
                                      epsilon=1e-8,use_locking=False,
                                      name='Adam').minimize(loss_blue_t)

"""################ Finish  build the separate model####################### """
print ('Finish  build the separate model!')
"""################ Start   train  the separate model####################### """
print ('start training.....')
##start train seperate channels :

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5

model_path="../model_sep"
if not os.path.isdir(model_path):
    os.mkdir(model_path)
"""##################                                     ####################"""

init=tf.global_variables_initializer()
saver=tf.train.Saver()
train_size=Red_data.shape[0]
train_indices=range(train_size)

#par_path="../par/"
#if not os.path.isdir(par_path):
#    os.mkdir(par_path)
with tf.Session(config=config) as sess:
    
    sess.run(init)
     
    for i in range(num_epoch):
        
        perm_indices=np.random.permutation(train_indices)
        
        for step in range(int(train_size/batch_size)):
             
            offset=(step*batch_size)%(train_size-batch_size)
            batch_indices=perm_indices[offset:(offset+batch_size)]

            feed_dict={
                       Red_input:Red_data[batch_indices,:],
                       Green_input:Green_data[batch_indices,:],
                       Blue_input:Blue_data[batch_indices,:]
                      }   
            
            #_red,l_red=sess.run([optimizer_red,loss_red],feed_dict=feed_dict)
            #_green,l_green=sess.run([optimizer_green,loss_green],feed_dict=feed_dict)
            _blue,l_blue=sess.run([optimizer_blue,blue_pixel_loss],feed_dict=feed_dict)
        #print('Depth loss is :',l_depth)
        #print('Depth regularization is :',rg)
        print('Blue loss is :',l_blue)
        #print('Building loss is :',l_Building)
    #saver.save(sess,model_path+"/pretraining_sep.ckpt")    
    print ('seperately pretraining finished')

