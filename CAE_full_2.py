import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from process_data import  process_data
tf.reset_default_graph()
batch_size=20
num_epochs=1
RESTORE=0
SEED = None
filter_size=16
rgbpath='./CNN_models/rgb_'+str(num_epochs)+'_'+str(filter_size)+'/rgb.ckpt'
depthpath='./CNN_models/depth_'+str(num_epochs)+'_'+str(filter_size)+'/depth.ckpt'
Sempath='./CNN_models/sem_'+str(num_epochs)+'_'+str(filter_size)+'/sem.ckpt'
fullpath='./CNN_models/full_'+str(num_epochs)+'_'+str(filter_size)+'/full.ckpt'


depth_data=np.load('../Data/depth_data.npy')
depth_mask=np.load('../Data/depth_mask.npy')
sem_data=np.load('../Data/sem_data.npy')
rgb_data=np.load('../Data/rgb_data.npy')
print(rgb_data.shape)
print(sem_data.shape)
print(depth_data.shape)
print(depth_mask.shape)


"data augmentation"
"depth"
"input"
depth_labels=np.concatenate((depth_data,depth_data),axis=0)
depth_data=np.concatenate((depth_data,0*depth_data),axis=0)
depth_mask=np.concatenate((depth_mask,depth_mask),axis=0)
"RGB"
rgb_data=np.concatenate((rgb_data,rgb_data),axis=0)
"Sem"
sem_labels=np.concatenate((sem_data,sem_data),axis=0)
sem_data=np.concatenate((sem_data,0*sem_data),axis=0)


print(rgb_data.shape)
print(sem_data.shape)
print(sem_labels.shape)
print(depth_data.shape)
print(depth_labels.shape)
print(depth_mask.shape)
"finish data augmentation"



with tf.variable_scope("Sem"):
    sem_inputs= tf.placeholder(tf.float32, (None, 18,60,5), name="sem_nput")
    sem_outputs=tf.placeholder(tf.float32, (None, 18,60,5), name="sem_ouput")
    ### Encoder use high level module 
    sem_conv1=tf.layers.conv2d(inputs=sem_inputs,filters=2*filter_size,kernel_size=(3,3),padding='same',
                       activation=tf.nn.relu,name='conv1')
#now (batch,18,60,16)

    sem_pool1=tf.layers.max_pooling2d(sem_conv1,pool_size=(2,2),strides=(2,2),padding='same')
#now (batch,9,30,16)

    sem_conv2=tf.layers.conv2d(inputs=sem_pool1,filters=filter_size,kernel_size=(3,3),padding='same',
                       activation=tf.nn.relu,name='conv2')
# now (batch,9,30,8)

    sem_pool2=tf.layers.max_pooling2d(sem_conv2,pool_size=(2,2),strides=(2,2),padding='same')
#now (batch,5,15,8)
with tf.variable_scope("RGB"):
    
    rgb_inputs=tf.placeholder(tf.float32, (None, 18,60,3), name="input")
    rgb_outputs=tf.placeholder(tf.float32, (None, 18,60,3), name="ouput")
    ### Encoder use high level module 
    rgb_conv1=tf.layers.conv2d(inputs=rgb_inputs,filters=2*filter_size,kernel_size=(3,3),padding='same',
                       activation=tf.nn.relu,name='conv1')
#now (batch,18,60,16)

    rgb_pool1=tf.layers.max_pooling2d(rgb_conv1,pool_size=(2,2),strides=(2,2),padding='same')
#now (batch,9,30,16)

    rgb_conv2=tf.layers.conv2d(inputs=rgb_pool1,filters=filter_size,kernel_size=(3,3),padding='same',
                       activation=tf.nn.relu,name='conv2')
# now (batch,9,30,8)

    rgb_pool2=tf.layers.max_pooling2d(rgb_conv2,pool_size=(2,2),strides=(2,2),padding='same')
#now (batch,5,15,8)

with tf.variable_scope("Depth"):
    depth_inputs=tf.placeholder(tf.float32, (None, 18,60,1), name="input")
    depth_outputs=tf.placeholder(tf.float32, (None, 18,60,1), name="ouput")
    depth_outmask=tf.placeholder(tf.float32, (None, 18,60,1), name="mask")
    ### Encoder use high level module 

    depth_conv1=tf.layers.conv2d(inputs=depth_inputs,filters=2*filter_size,kernel_size=(3,3),padding='same',
                       activation=tf.nn.relu,name='conv1')
#now (batch,18,60,16)

    depth_pool1=tf.layers.max_pooling2d(depth_conv1,pool_size=(2,2),strides=(2,2),padding='same')
#now (batch,9,30,16)

    depth_conv2=tf.layers.conv2d(inputs=depth_pool1,filters=filter_size,kernel_size=(3,3),padding='same',
                       activation=tf.nn.relu,name='conv2')
# now (batch,9,30,8)

    depth_pool2=tf.layers.max_pooling2d(depth_conv2,pool_size=(2,2),strides=(2,2),padding='same')
#now (batch,5,15,8)



with tf.variable_scope("full"):
    full_in=tf.concat([rgb_pool2,depth_pool2,sem_pool2],axis=3)
#now (batch,5,15,24)
    full_shared=tf.layers.conv2d(inputs=full_in,filters=filter_size,kernel_size=(3,3),padding='same',
                                 activation=tf.nn.relu,name='full_shared')
#now (batch,5,15,8)
    full_out=tf.layers.conv2d(inputs=full_shared,filters=3*filter_size,kernel_size=(3,3),padding='same',
                              activation=tf.nn.relu,name='full_out')
#now (batch,5,15,24)
    rgb_pool2_out,depth_pool2_out,sem_pool2_out=tf.split(full_out,3,axis=3)



with tf.variable_scope("Sem"):
#########################################################################################
    ### Decoder using high level modules 
    sem_upsample1=tf.image.resize_images(sem_pool2_out,size=(9,30),
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# now (batch,9,30,8)
    sem_conv4=tf.layers.conv2d(inputs=sem_upsample1,filters=2*filter_size,kernel_size=(3,3),padding='same',
                       activation=tf.nn.relu,name='conv4')
#now (batch,9,30,8)

    sem_upsample2 = tf.image.resize_images(sem_conv4, size=(18,60),
                                   method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
#now (batch,18,60,8)
    sem_out=tf.layers.conv2d(inputs=sem_upsample2,filters=5,kernel_size=(3,3),padding='same',
                       activation=tf.nn.sigmoid,name='out')
#now (batch,18,60,3)



with tf.variable_scope("RGB"):
#########################################################################################
    ### Decoder using high level modules 
    rgb_upsample1=tf.image.resize_images(rgb_pool2_out,size=(9,30),
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# now (batch,9,30,8)
    rgb_conv4=tf.layers.conv2d(inputs=rgb_upsample1,filters=2*filter_size,kernel_size=(3,3),padding='same',
                       activation=tf.nn.relu,name='conv4')
#now (batch,9,30,8)

    rgb_upsample2 = tf.image.resize_images(rgb_conv4, size=(18,60),
                                   method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
#now (batch,18,60,8)
    rgb_out=tf.layers.conv2d(inputs=rgb_upsample2,filters=3,kernel_size=(3,3),padding='same',
                       activation=tf.nn.sigmoid,name='out')
#now (batch,18,60,3)


with tf.variable_scope("Depth"):
#########################################################################################
    ### Decoder using high level modules 
    depth_upsample1=tf.image.resize_images(depth_pool2_out,size=(9,30),
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# now (batch,9,30,8)
    depth_conv4=tf.layers.conv2d(inputs=depth_upsample1,filters=2*filter_size,kernel_size=(3,3),padding='same',
                       activation=tf.nn.relu,name='conv4')
#now (batch,9,30,8)

    depth_upsample2 = tf.image.resize_images(depth_conv4, size=(18,60),
                                   method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
#now (batch,18,60,8)
    depth_out=tf.layers.conv2d(inputs=depth_upsample2,filters=1,kernel_size=(3,3),padding='same',
                       activation=tf.nn.relu,name='out')
#now (batch,18,60,3)



var_sem=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='Sem')
var_rgb=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='RGB')
var_depth=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='Depth')
var_full=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='full')

learning_rate=1e-4
loss=(tf.nn.l2_loss( (depth_out-depth_outputs)*depth_outmask )
      +tf.nn.l2_loss(sem_out-sem_outputs)
      +tf.nn.l2_loss(rgb_out-rgb_outputs)
      )
optimizer1=tf.train.AdamOptimizer(learning_rate).minimize(loss,var_list=var_full)
optimizer2=tf.train.AdamOptimizer(learning_rate=1e-6).minimize(loss)


saver_sem=tf.train.Saver(var_sem)
saver_depth=tf.train.Saver(var_depth)
saver_rgb=tf.train.Saver(var_rgb)
saver_full=tf.train.Saver()


init=tf.global_variables_initializer()
train_size=rgb_data.shape[0]
train_indices=range(train_size)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction =0.4


with tf.Session(config=config) as sess:
    sess.run(init)
    saver_sem.restore(sess,Sempath)
    saver_depth.restore(sess,depthpath)
    saver_rgb.restore(sess,rgbpath)


    for ipochs in range(int(1+0.6*num_epochs)):
        perm_indices=np.random.permutation(train_indices)

        for step in range(int(train_size/batch_size)):
            offset=(step*batch_size)%(train_size-batch_size)
            batch_indices=perm_indices[offset:(offset+batch_size)]
            l,_=sess.run([loss,optimizer1],feed_dict={sem_inputs:sem_data[batch_indices],
                                                     sem_outputs:sem_data[batch_indices],
                                                     depth_inputs:depth_data[batch_indices],
                                                     depth_outputs:depth_data[batch_indices],
                                                     depth_outmask:depth_mask[batch_indices],
                                                     rgb_inputs:rgb_data[batch_indices],
                                                     rgb_outputs:rgb_data[batch_indices]
                                                     })
        print("Epoch: {}...".format(ipochs),
                       "Training loss: {:.4f}".format(l))

    for ipochs in range(int(1+0.4*num_epochs)):
        perm_indices=np.random.permutation(train_indices)

        for step in range(int(train_size/batch_size)):
            offset=(step*batch_size)%(train_size-batch_size)
            batch_indices=perm_indices[offset:(offset+batch_size)]
            l,_=sess.run([loss,optimizer2],feed_dict={sem_inputs:sem_data[batch_indices],
                                                     sem_outputs:sem_labels[batch_indices],
                                                     depth_inputs:depth_data[batch_indices],
                                                     depth_outputs:depth_labels[batch_indices],
                                                     depth_outmask:depth_mask[batch_indices],
                                                     rgb_inputs:rgb_data[batch_indices],
                                                     rgb_outputs:rgb_data[batch_indices]
                                                     })
        print("Epoch: {}...".format(ipochs),
                       "Training loss: {:.4f}".format(l))

        saver_full.save(sess,fullpath)

