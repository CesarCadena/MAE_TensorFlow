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
    Depth_data=data['Depth']
    Depthmask_data=data['Depthmask']
    Ground_data=data['Ground']
    Objects_data=data['Objects']
    Building_data=data['Building']
    Vegetation_data=data['Vegetation']
    Sky_data=data['Sky']


Depth_data=np.multiply(Depth_data,Depthmask_data)
#u_depth=Depth_data.mean(0)
#Depth_data=Depth_data-u_depth



print ("Finish loading the data !")


"""################ Start build the separate model####################### """

"""RGB Depth model:"""
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
loss_blue=tf.nn.l2_loss(Blue_input-Blue_out)

regularization_blue=tf.nn.l2_loss(Blue_weights)+tf.nn.l2_loss(Blue_outweights)

loss_blue_t=loss_blue+1e-4*regularization_blue

optimizer_blue=tf.train.AdamOptimizer(learning_rate=1e-4,
                                      beta1=0.9,beta2=0.999,
                                      epsilon=1e-8,use_locking=False,
                                      name='Adam').minimize(loss_blue_t)


""" depth channel"""
Depth_input=tf.placeholder(tf.float32,shape=[None,1080])
#Depthmask_input=tf.placeholder(tf.float32,shape=[None,1080])
Depth_weights=tf.Variable(tf.random_normal(shape=[1080,hidden_size],
                                  stddev=0.01),name="Depth_weights")
Depth_bias=tf.Variable(tf.zeros([1,hidden_size]),name="Depth_bias")
Depth_hidden=tf.nn.relu(tf.matmul(Depth_input,Depth_weights)+Depth_bias)


Depth_outweights=tf.Variable(tf.random_normal(shape=[hidden_size,1080],
                                  stddev=0.01),name="Depth_outweights")
Depth_outbias=tf.Variable(tf.zeros([1,1080]),name="Depth_outbias")
Depth_out=tf.matmul(Depth_hidden,Depth_outweights)+Depth_outbias

loss_depth=tf.nn.l2_loss(Depth_out-Depth_input)


regularization_depth=tf.nn.l2_loss(Depth_weights)+tf.nn.l2_loss(Depth_outweights)
loss_depth_t=loss_depth+1e-4*regularization_depth

optimizer_depth=tf.train.AdamOptimizer(learning_rate=1e-5,
                                      beta1=0.9,beta2=0.999,
                                      epsilon=1e-8,use_locking=False,
                                      name='Adam').minimize(loss_depth_t)


""" Ground channel """
Ground_input=tf.placeholder(tf.float32,shape=[None,1080])
Ground_weights=tf.Variable(tf.random_normal(shape=[1080,hidden_size],
                                   stddev=0.01),name="Ground_weights")
Ground_bias=tf.Variable(tf.zeros([1,hidden_size]),name="Ground_bias")
Ground_hidden=tf.nn.relu(tf.matmul(Ground_input,Ground_weights)+Ground_bias)

Ground_outweights=tf.Variable(tf.random_normal(shape=[hidden_size,1080],
                                   stddev=0.01),name="Ground_outweights")

Ground_outbias=tf.Variable(tf.zeros([1,1080]),name="Ground_outbias")
Ground_out=tf.nn.sigmoid(tf.matmul(Ground_hidden,Ground_outweights)+Ground_outbias)

loss_Ground=-tf.reduce_sum(np.multiply(Ground_input,tf.log(tf.clip_by_value(Ground_out,1e-10,1)))
                          +np.multiply(1-Ground_input,tf.log(tf.clip_by_value(1-Ground_out,1e-10,1)))
                          )

regularization_Ground=tf.nn.l2_loss(Ground_weights)+tf.nn.l2_loss(Ground_outweights)
loss_Ground_t=loss_Ground+1e-4*regularization_Ground
optimizer_Ground=tf.train.AdamOptimizer(learning_rate=1e-4,
                                      beta1=0.9,beta2=0.999,
                                      epsilon=1e-8,use_locking=False,
                                      name='Adam').minimize(loss_Ground_t)

"""Objects channel"""

Objects_input=tf.placeholder(tf.float32,shape=[None,1080])
Objects_weights=tf.Variable(tf.random_normal(shape=[1080,hidden_size],
                                   stddev=0.01),name="Objects_weights")
Objects_bias=tf.Variable(tf.zeros([1,hidden_size]),name="Objects_bias")

Objects_hidden=tf.nn.relu(tf.matmul(Objects_input,Objects_weights)+Objects_bias)
Objects_outweights=tf.Variable(tf.random_normal(shape=[hidden_size,1080],
                                    stddev=0.01),name="Objects_outweights")
Objects_outbias=tf.Variable(tf.zeros([1,1080]),name="Objects_outbias")

Objects_out=tf.nn.sigmoid(tf.matmul(Objects_hidden,Objects_outweights)+Objects_outbias)


#loss_Objects=tf.nn.l2_loss(Objects_input-Objects_out)
loss_Objects=-tf.reduce_sum(np.multiply(Objects_input,tf.log(tf.clip_by_value(Objects_out,1e-10,1)))
                          +np.multiply(1-Objects_input,tf.log(tf.clip_by_value(1-Objects_out,1e-10,1)))
                          )
regularization_Objects=tf.nn.l2_loss(Objects_weights)+tf.nn.l2_loss(Objects_outweights)
             

loss_Objects_t=loss_Objects+1e-4*regularization_Objects

optimizer_Objects=tf.train.AdamOptimizer(learning_rate=1e-4,
                                      beta1=0.9,beta2=0.999,
                                      epsilon=1e-8,use_locking=False,
                                      name='Adam').minimize(loss_Objects_t)

"""Building channels"""
Building_input=tf.placeholder(tf.float32,shape=[None,1080])
Building_weights=tf.Variable(tf.random_normal(shape=[1080,hidden_size],
                                   stddev=0.01),name="Building_weights")
Building_bias=tf.Variable(tf.zeros([1,hidden_size]),name="Building_bias")

Building_hidden=tf.nn.relu(tf.matmul(Building_input,Building_weights)+Building_bias)

Building_outweights=tf.Variable(tf.random_normal(shape=[hidden_size,1080],
                                    stddev=0.01),name="Building_outweights")
Building_outbias=tf.Variable(tf.zeros([1,1080]),name="Building_outbias")
Building_out=tf.nn.sigmoid(tf.matmul(Building_hidden,Building_outweights)+Building_outbias)


loss_Building=-tf.reduce_sum(np.multiply(Building_input,tf.log(tf.clip_by_value(Building_out,1e-10,1)))
                          +np.multiply(1-Building_input,tf.log(tf.clip_by_value(1-Building_out,1e-10,1)))
                          )

regularization_Building=tf.nn.l2_loss(Building_weights)+tf.nn.l2_loss(Building_outweights)
loss_Building_t=loss_Building+1e-4*regularization_Building

optimizer_Building=tf.train.AdamOptimizer(learning_rate=1e-4,
                                      beta1=0.9,beta2=0.999,
                                      epsilon=1e-8,use_locking=False,
                                      name='Adam').minimize(loss_Building_t)

"""Sky channels"""
Sky_input=tf.placeholder(tf.float32,shape=[None,1080])
Sky_weights=tf.Variable(tf.random_normal(shape=[1080,hidden_size],
                                   stddev=0.01),name="Sky_weights")
Sky_bias=tf.Variable(tf.zeros([1,hidden_size]),name="Sky_bias")

Sky_hidden=tf.nn.relu(tf.matmul(Sky_input,Sky_weights)+Sky_bias)
Sky_outweights=tf.Variable(tf.random_normal(shape=[hidden_size,1080],
                                   stddev=0.01),name="Sky_outweights")
Sky_outbias=tf.Variable(tf.zeros([1,1080]),name="Sky_outbias")
Sky_out=tf.nn.sigmoid(tf.matmul(Sky_hidden,Sky_outweights)+Sky_outbias)
#loss_Sky=tf.nn.l2_loss(Sky_input-Sky_out)
loss_Sky=-tf.reduce_sum(np.multiply(Sky_input,tf.log(tf.clip_by_value(Sky_out,1e-10,1)))
                          +np.multiply(1-Sky_input,tf.log(tf.clip_by_value(1-Sky_out,1e-10,1)))
                          )

regularization_Sky=tf.nn.l2_loss(Sky_weights)+tf.nn.l2_loss(Sky_outweights)


loss_Sky_t=loss_Sky+1e-4*regularization_Sky
optimizer_Sky=tf.train.AdamOptimizer(learning_rate=1e-4,
                                      beta1=0.9,beta2=0.999,
                                      epsilon=1e-8,use_locking=False,
                                      name='Adam').minimize(loss_Sky_t)

"""channel Vegetation"""
                                                                                      
Vegetation_input=tf.placeholder(tf.float32,shape=[None,1080])

Vegetation_weights=tf.Variable(tf.random_normal(shape=[1080,hidden_size],
                                   stddev=0.01),name="Vegetation_weights")

Vegetation_bias=tf.Variable(tf.zeros([1,hidden_size]),name="Vegetation_bias")
Vegetation_hidden=tf.nn.relu(tf.matmul(Vegetation_input,Vegetation_weights)+Vegetation_bias)

Vegetation_outweights=tf.Variable(tf.random_normal(shape=[hidden_size,1080],
                                    stddev=0.01),name="Vegetation_outweights")
Vegetation_outbias=tf.Variable(tf.zeros([1,1080]),name="Vegetation_outbias")
Vegetation_out=tf.nn.sigmoid(tf.matmul(Vegetation_hidden,Vegetation_outweights)+Vegetation_outbias)
#loss_Vegetation=tf.nn.l2_loss(Vegetation_input-Vegetation_out)

loss_Vegetation=-tf.reduce_sum(np.multiply(Vegetation_input,tf.log(tf.clip_by_value(Vegetation_out,1e-10,1)))
                          +np.multiply(1-Vegetation_input,tf.log(tf.clip_by_value(1-Vegetation_out,1e-10,1)))
                          )

regularization_Vegetation=tf.nn.l2_loss(Vegetation_weights)+tf.nn.l2_loss(Vegetation_outweights)

loss_Vegetation_t=loss_Vegetation+1e-4*regularization_Vegetation

optimizer_Vegetation=tf.train.AdamOptimizer(learning_rate=1e-4,
                                      beta1=0.9,beta2=0.999,
                                      epsilon=1e-8,use_locking=False,
                                      name='Adam').minimize(loss_Vegetation_t)

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

            feed_dict={Ground_input:Ground_data[batch_indices,:],
                       Objects_input:Objects_data[batch_indices,:],
                       Vegetation_input:Vegetation_data[batch_indices,:],
                       Building_input:Building_data[batch_indices,:],
                       Sky_input:Sky_data[batch_indices,:],
                       Red_input:Red_data[batch_indices,:],
                       Green_input:Green_data[batch_indices,:],
                       Blue_input:Blue_data[batch_indices,:],
                       Depth_input:Depth_data[batch_indices,:]
                       #Depthmask_input:Depthmask_data[batch_indices,:]
                      }   
            

            _red,l_red=sess.run([optimizer_red,loss_red],feed_dict=feed_dict)
            
            _green,l_green=sess.run([optimizer_green,loss_green],feed_dict=feed_dict)
            
            _blue,l_blue=sess.run([optimizer_blue,loss_blue],feed_dict=feed_dict)
            
            _depth,l_depth,rg=sess.run([optimizer_depth,loss_depth,regularization_depth],feed_dict=feed_dict)

            _Ground,l_Ground=sess.run([optimizer_Ground,loss_Ground],feed_dict=feed_dict)

            _Objects,l_Objects=sess.run([optimizer_Objects,loss_Objects],feed_dict=feed_dict)

            _Building,l_Building=sess.run([optimizer_Building,loss_Building],feed_dict=feed_dict)

            _Vegetation,l_Vegetation=sess.run([optimizer_Vegetation,loss_Vegetation],feed_dict=feed_dict)

            _Sky,l_Sky=sess.run([optimizer_Sky,loss_Sky],feed_dict=feed_dict)

        print('Depth loss is :',l_depth)
        print('Depth regularization is :',rg)
        print('Blue loss is :',l_blue)
        print('Building loss is :',l_Building)

        

    saver.save(sess,model_path+"/pretraining_sep.ckpt")    
    print ('seperately pretraining finished')

########################################
# Next start train the full shared MAE
########################################
