# Draft MAE Model in tensorflow based on Dr.Cesar Cadena
# this code is developed by Yi Liu 
#run the code under the folder /Desktop/project
#be carful this version works but not converged yet !!!

import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from tensorflow.examples.tutorials.mnist import input_data
batch_size=128
num_epochs=1
learning_rate=1e-6
hidden_size=1024
NOTRESTORE=1


project_dir='/Users/yi/Desktop/Desktop/project/MAE'
tf.app.flags.DEFINE_string('train_dir',project_dir,"where to write checkpoints and save graphs")
FLAGS = tf.app.flags.FLAGS


###build the model 
# totally 9 channels 
Red_input=tf.placeholder(tf.float32,shape=[batch_size,1080])
Blue_input=tf.placeholder(tf.float32,shape=[batch_size,1080])
Green_input=tf.placeholder(tf.float32,shape=[batch_size,1080])
Depth_input=tf.placeholder(tf.float32,shape=[batch_size,1080])
Ground_input=tf.placeholder(tf.float32,shape=[batch_size,1080])
Objects_input=tf.placeholder(tf.float32,shape=[batch_size,1080])
Building_input=tf.placeholder(tf.float32,shape=[batch_size,1080])
Vegetation_input=tf.placeholder(tf.float32,shape=[batch_size,1080])
Sky_input=tf.placeholder(tf.float32,shape=[batch_size,1080])


Depth_weights=tf.Variable(tf.random_normal(shape=[1080,hidden_size],
                                  stddev=0.01),name="Depth_weights")
Depth_bias=tf.Variable(tf.zeros([1,hidden_size]),name="Depth_bias")


Red_weights=tf.Variable(tf.random_normal(shape=[1080,hidden_size],
                                  stddev=0.01),name="Red_weights")
Red_bias=tf.Variable(tf.zeros([1,hidden_size]),name="Red_bias")


Blue_weights=tf.Variable(tf.random_normal(shape=[1080,hidden_size],
                                   stddev=0.01),name="Blue_weights")
Blue_bias=tf.Variable(tf.zeros([1,hidden_size]),name="Blue_bias")


Green_weights=tf.Variable(tf.random_normal(shape=[1080,hidden_size],
                                   stddev=0.01),name="Green_weights")
Green_bias=tf.Variable(tf.zeros([1,hidden_size]),name="Green_bias")


Ground_weights=tf.Variable(tf.random_normal(shape=[1080,hidden_size],
                                   stddev=0.01),name="Ground_weights")
Ground_bias=tf.Variable(tf.zeros([1,hidden_size]),name="Ground_bias")


Objects_weights=tf.Variable(tf.random_normal(shape=[1080,hidden_size],
                                   stddev=0.01),name="Objects_weights")
Objects_bias=tf.Variable(tf.zeros([1,hidden_size]),name="Objects_bias")


Building_weights=tf.Variable(tf.random_normal(shape=[1080,hidden_size],
                                   stddev=0.01),name="Building_weights")
Building_bias=tf.Variable(tf.zeros([1,hidden_size]),name="Building_bias")


Vegetation_weights=tf.Variable(tf.random_normal(shape=[1080,hidden_size],
                                   stddev=0.01),name="Vegetation_weights")
Vegetation_bias=tf.Variable(tf.zeros([1,hidden_size]),name="Vegetation_bias")


Sky_weights=tf.Variable(tf.random_normal(shape=[1080,hidden_size],
                                   stddev=0.01),name="Sky_weights")
Sky_bias=tf.Variable(tf.zeros([1,hidden_size]),name="Sky_bias")


Depth_hidden=tf.nn.relu(tf.matmul(Depth_input,Depth_weights)+Depth_bias)
Red_hidden=tf.nn.relu(tf.matmul(Red_input,Red_weights)+Red_bias)
Blue_hidden=tf.nn.relu(tf.matmul(Blue_input,Blue_weights)+Blue_bias)
Green_hidden=tf.nn.relu(tf.matmul(Green_input,Green_weights)+Green_bias)
Ground_hidden=tf.nn.relu(tf.matmul(Ground_input,Ground_weights)+Ground_bias)
Objects_hidden=tf.nn.relu(tf.matmul(Objects_input,Objects_weights)+Objects_bias)
Building_hidden=tf.nn.relu(tf.matmul(Building_input,Building_weights)+Building_bias)
Vegetation_hidden=tf.nn.relu(tf.matmul(Vegetation_input,Vegetation_weights)+Vegetation_bias)
Sky_hidden=tf.nn.relu(tf.matmul(Sky_input,Sky_weights)+Sky_bias)


Semantic_weights=tf.Variable(tf.random_normal(shape=[5*hidden_size,hidden_size],
                             stddev=0.01),name="Semantic_weights")
Semantic_bias=tf.Variable(tf.zeros([1,hidden_size]),name="Semantic_bias")

Semantic_shared=tf.nn.relu(tf.matmul(tf.concat([Ground_hidden,Objects_hidden,Building_hidden,Vegetation_hidden,Sky_hidden],1)
                                     ,Semantic_weights)+Semantic_bias)



Fullshared_weights=tf.Variable(tf.random_normal(shape=[5*hidden_size,hidden_size],
                               stddev=0.01),name="Fullshared_weights")
Fullshared_bias=tf.Variable(tf.zeros([1,hidden_size]),name="Fullshared_bias")

Full_shared=tf.nn.relu(tf.matmul(tf.concat([Depth_hidden,Red_hidden,Green_hidden,Blue_hidden,
                                            Semantic_shared],1),Fullshared_weights)+Fullshared_bias)





#####decoder 

decoder_weights=tf.Variable(tf.random_normal(shape=[hidden_size,5*hidden_size],
                           stddev=0.01),name="decoder1_weights")
decoder_bias=tf.Variable(tf.zeros([1,5*hidden_size]),name="decoder1_bias")

decoder_layer1=tf.nn.relu(tf.matmul(Full_shared,decoder_weights)+decoder_bias)


decoder_Depth,decoder_Red,decoder_Green,decoder_Blue,decoder_sem1=tf.split(decoder_layer1,num_or_size_splits=5, axis=1)


decoder_semweights=tf.Variable(tf.random_normal(shape=[hidden_size,5*hidden_size],stddev=0.01),name="decoder_semweights")
decoder_sembias=tf.Variable(tf.zeros([1,5*hidden_size]),name="decoder_sembias")
decoder_sem2=tf.nn.relu(tf.matmul(decoder_sem1,decoder_semweights)+decoder_sembias)


decoder_ground,decoder_objects,decoder_building,decoder_vegetation,decoder_sky=tf.split(decoder_sem2,num_or_size_splits=5, axis=1)



Depth_outweights=tf.Variable(tf.random_normal(shape=[hidden_size,1080],stddev=0.01),name="Depth_outweights")
Depth_outbias=tf.Variable(tf.zeros([1,1080]),name="Depth_outbias")
Depth_out=tf.nn.relu(tf.matmul(decoder_Depth,Depth_outweights)+Depth_outbias)

Red_outweights=tf.Variable(tf.random_normal(shape=[hidden_size,1080],stddev=0.01),name="Red_outweights")
Red_outbias=tf.Variable(tf.zeros([1,1080]),name="Red_outbias")
Red_out=tf.nn.sigmoid(tf.matmul(decoder_Red,Red_outweights)+Red_outbias)

Green_outweights=tf.Variable(tf.random_normal(shape=[hidden_size,1080],stddev=0.01),name="Green_outweights")
Green_outbias=tf.Variable(tf.zeros([1,1080]),name="Green_outbias")
Green_out=tf.nn.sigmoid(tf.matmul(decoder_Green,Green_outweights)+Green_outbias)

Blue_outweights=tf.Variable(tf.random_normal(shape=[hidden_size,1080],stddev=0.01),name="Blue_outweights")
Blue_outbias=tf.Variable(tf.zeros([1,1080]),name="Blue_outbias")
Blue_out=tf.nn.sigmoid(tf.matmul(decoder_Blue,Blue_outweights)+Blue_outbias)


Ground_outweights=tf.Variable(tf.random_normal(shape=[hidden_size,1080],stddev=0.01),name="ground_outweights")
Ground_outbias=tf.Variable(tf.zeros([1,1080]),name="ground_outbias")
Ground_out=tf.nn.relu(tf.matmul(decoder_ground,Ground_outweights)+Ground_outbias)

Objects_outweights=tf.Variable(tf.random_normal(shape=[hidden_size,1080],stddev=0.01),name="Objects_outweights")
Objects_outbias=tf.Variable(tf.zeros([1,1080]),name="Objects_outbias")
Objects_out=tf.nn.relu(tf.matmul(decoder_objects,Objects_outweights)+Objects_outbias)


Building_outweights=tf.Variable(tf.random_normal(shape=[hidden_size,1080],stddev=0.01),name="Building_outweights")
Building_outbias=tf.Variable(tf.zeros([1,1080]),name="Building_outbias")
Building_out=tf.nn.relu(tf.matmul(decoder_building,Building_outweights)+Building_outbias)


Vegetation_outweights=tf.Variable(tf.random_normal(shape=[hidden_size,1080],stddev=0.01),name="vegetation_outweights")
Vegetation_outbias=tf.Variable(tf.zeros([1,1080]),name="Vegetation_outbias")
Vegetation_out=tf.nn.relu(tf.matmul(decoder_vegetation,Vegetation_outweights)+Vegetation_outbias)

Sky_outweights=tf.Variable(tf.random_normal(shape=[hidden_size,1080],stddev=0.01),name="Sky_outweights")
Sky_outbias=tf.Variable(tf.zeros([1,1080]),name="Sky_outbias")
Sky_out=tf.nn.relu(tf.matmul(decoder_sky,Sky_outweights)+Sky_outbias)




#  train the model 

init=tf.global_variables_initializer()

# be careful with this loss function

loss=(tf.nn.l2_loss(Red_input-Red_out)
     +tf.nn.l2_loss(Blue_input-Blue_out)
     +tf.nn.l2_loss(Green_input-Green_out)
     +tf.nn.l2_loss(Depth_input-Depth_out)
     +tf.nn.l2_loss(Ground_input-Ground_out)
     +tf.nn.l2_loss(Objects_input-Objects_out)
     +tf.nn.l2_loss(Building_input-Building_out)
     +tf.nn.l2_loss(Vegetation_input-Vegetation_out)
     +tf.nn.l2_loss(Sky_input-Sky_out))/batch_size



regularizers=(tf.nn.l2_loss(Depth_weights)+tf.nn.l2_loss(Depth_outbias)
             +tf.nn.l2_loss(Red_weights)+tf.nn.l2_loss(Red_bias)
             +tf.nn.l2_loss(Blue_weights)+tf.nn.l2_loss(Blue_bias)
             +tf.nn.l2_loss(Green_weights)+tf.nn.l2_loss(Green_bias)
             +tf.nn.l2_loss(Ground_weights)+tf.nn.l2_loss(Ground_bias)
             +tf.nn.l2_loss(Objects_weights)+tf.nn.l2_loss(Objects_bias)
             +tf.nn.l2_loss(Building_weights)+tf.nn.l2_loss(Building_bias)
             +tf.nn.l2_loss(Vegetation_weights)+tf.nn.l2_loss(Vegetation_bias)
             +tf.nn.l2_loss(Sky_weights)+tf.nn.l2_loss(Sky_bias)
             +tf.nn.l2_loss(Semantic_weights)+tf.nn.l2_loss(Semantic_bias)
             +tf.nn.l2_loss(Fullshared_weights)+tf.nn.l2_loss(Fullshared_bias)
             +tf.nn.l2_loss(decoder_weights)+tf.nn.l2_loss(decoder_bias)
             +tf.nn.l2_loss(decoder_semweights)+tf.nn.l2_loss(decoder_sembias)
             +tf.nn.l2_loss(Depth_outweights)+tf.nn.l2_loss(Depth_outbias)
             +tf.nn.l2_loss(Red_outweights)+tf.nn.l2_loss(Red_outbias)
             +tf.nn.l2_loss(Blue_outweights)+tf.nn.l2_loss(Blue_outbias)
             +tf.nn.l2_loss(Green_outweights)+tf.nn.l2_loss(Green_outbias)
             +tf.nn.l2_loss(Ground_outweights)+tf.nn.l2_loss(Green_outbias)
             +tf.nn.l2_loss(Objects_outweights)+tf.nn.l2_loss(Objects_outbias)
             +tf.nn.l2_loss(Building_outweights)+tf.nn.l2_loss(Building_outbias)
             +tf.nn.l2_loss(Vegetation_outweights)+tf.nn.l2_loss(Vegetation_outbias)
             +tf.nn.l2_loss(Sky_outweights)+tf.nn.l2_loss(Sky_outbias))

              
loss=loss+regularizers

tf.summary.scalar('loss',loss)

optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

saver=tf.train.Saver()



#  prepare data 
import pandas as pd
import tensorflow
import scipy.io
foldernamme='MAE_KITTI/data_18x60/'
training_index=pd.read_csv('MAE_KITTI/kitti_split.txt',header=None)
training_index=training_index[1:29]
training_index=training_index.values

xcr=np.empty((1080,0))
xcg=np.empty((1080,0))
xcb=np.empty((1080,0))
xid=np.empty((1080,0))
Ground=np.empty((1080,0))
Objects=np.empty((1080,0))
Building=np.empty((1080,0))
Vegetation=np.empty((1080,0))
Sky=np.empty((1080,0))



for i in range(28):
    name=training_index[i][0]
    RGBname=foldernamme+'data_kitti_im02_'+name+'_18x60.mat'
    Depthname=foldernamme+'data_kitti_InvDepth02_'+name+'_18x60'
    Semname=foldernamme+'data_kitti_seg02_'+name+'_18x60'

    mat=scipy.io.loadmat(RGBname)
    xcr=np.append(xcr,mat['xcr'],axis=1)
    xcg=np.append(xcg,mat['xcg'],axis=1)
    xcb=np.append(xcb,mat['xcb'],axis=1)

    matd=scipy.io.loadmat(Depthname)
    xid=np.append(xid,matd['xid'],axis=1)

    mats=scipy.io.loadmat(Semname)

    Ground=np.append(Ground,(mats['xss']==1).astype(int),axis=1)
    Objects=np.append(Objects,(mats['xss']==2).astype(int),axis=1)
    Building=np.append(Building,(mats['xss']==3).astype(int),axis=1)
    Vegetation=np.append(Vegetation,(mats['xss']==4).astype(int),axis=1)
    Sky=np.append(Sky,(mats['xss']==5).astype(int),axis=1)



xcr=np.transpose(xcr)
xcg=np.transpose(xcg)
xcb=np.transpose(xcb)
xid=np.transpose(xid)
Ground=np.transpose(Ground)
Objects=np.transpose(Objects)
Building=np.transpose(Building)
Vegetation=np.transpose(Vegetation)
Sky=np.transpose(Sky)


print(xcr.shape)
print(xcg.shape)
print(xcb.shape)
print(xid.shape)
print(Ground.shape)
print(Objects.shape)
print(Building.shape)
print(Vegetation.shape)
print(Sky.shape)



train_size=int(xcr.shape[0])
train_indices=range(train_size)

with tf.Session() as sess:

    tf.global_variables_initializer().run()

    #summary_op=tf.summary.merge_all()
    #summary_writer=tf.summary.FileWriter(FLAGS.train_dir,graph=sess.graph)

    for ipoch in range(num_epochs):

        perm_indices=np.random.permutation(train_indices)

        for step in range(int(train_size/batch_size)):

            offset=(step*batch_size)%(train_size-batch_size)
            batch_indices=perm_indices[offset:(offset+batch_size)]

            feed_dict={Red_input:xcr[batch_indices,:],
                       Green_input:xcg[batch_indices,:],
                       Blue_input:xcb[batch_indices,:],
                       Depth_input:xid[batch_indices,:],
                       Ground_input:Ground[batch_indices,:],
                       Objects_input:Objects[batch_indices,:],
                       Building_input:Building[batch_indices,:],
                       Vegetation_input:Vegetation[batch_indices,:],
                       Sky_input:Sky[batch_indices,:]}

            _,l=sess.run([optimizer,loss],feed_dict=feed_dict)
            print ('loss is ' ,l)

        save_path=saver.save(sess,FLAGS.train_dir+"/model.ckpt")
        print("loss is :%d\n." % l)
        print("Model saved in file: %s" % save_path)

