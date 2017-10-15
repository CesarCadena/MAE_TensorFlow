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
num_epoch=200



FLAG='train'
if (FLAG=='train'):
    data=process_data('training')
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



pre_Ground_weights=np.load("../par/Ground_weights.npy")
pre_Ground_bias=np.load("../par/Ground_bias.npy")
pre_Ground_outweights=np.load("../par/Ground_outweights.npy")
pre_Ground_outbias=np.load("../par/Ground_outbias.npy")

pre_Objects_weights=np.load("../par/Objects_weights.npy")
pre_Objects_bias=np.load("../par/Objects_bias.npy")
pre_Objects_outweights=np.load("../par/Objects_outweights.npy")
pre_Objects_outbias=np.load("../par/Objects_outbias.npy")

pre_Building_weights=np.load("../par/Building_weights.npy")
pre_Building_bias=np.load("../par/Building_bias.npy")
pre_Building_outweights=np.load("../par/Building_outweights.npy")
pre_Building_outbias=np.load("../par/Building_outbias.npy")

pre_Vegetation_weights=np.load("../par/Vegetation_weights.npy")
pre_Vegetation_bias=np.load("../par/Vegetation_bias.npy")
pre_Vegetation_outweights=np.load("../par/Vegetation_outweights.npy")
pre_Vegetation_outbias=np.load("../par/Vegetation_outbias.npy")

pre_Sky_weights=np.load("../par/Sky_weights.npy")
pre_Sky_bias=np.load("../par/Sky_bias.npy")
pre_Sky_outweights=np.load("../par/Sky_outweights.npy")
pre_Sky_outbias=np.load("../par/Sky_outbias.npy")





# shared semantic model 
Ground_input=tf.placeholder(tf.float32,shape=[None,1080])
Objects_input=tf.placeholder(tf.float32,shape=[None,1080])
Building_input=tf.placeholder(tf.float32,shape=[None,1080])
Vegetation_input=tf.placeholder(tf.float32,shape=[None,1080])
Sky_input=tf.placeholder(tf.float32,shape=[None,1080])




Ground_weights=tf.Variable(pre_Ground_weights,name="Ground_weights")
Ground_bias=tf.Variable(pre_Ground_bias,name="Ground_bias")


Objects_weights=tf.Variable(pre_Objects_weights,name="Objects_weights")
Objects_bias=tf.Variable(pre_Objects_bias,name="Objects_bias")


Building_weights=tf.Variable(pre_Building_weights,name="Building_weights")
Building_bias=tf.Variable(pre_Building_bias,name="Building_bias")


Vegetation_weights=tf.Variable(pre_Vegetation_weights,name="Vegetation_weights")
Vegetation_bias=tf.Variable(pre_Vegetation_bias,name="Vegetation_bias")



Sky_weights=tf.Variable(pre_Sky_weights,name="Sky_weights")
Sky_bias=tf.Variable(pre_Sky_bias,name="Sky_bias")


Ground_hidden=tf.nn.relu(tf.matmul(Ground_input,Ground_weights)+Ground_bias)
Objects_hidden=tf.nn.relu(tf.matmul(Objects_input,Objects_weights)+Objects_bias)
Building_hidden=tf.nn.relu(tf.matmul(Building_input,Building_weights)+Building_bias)
Vegetation_hidden=tf.nn.relu(tf.matmul(Vegetation_input,Vegetation_weights)+Vegetation_bias)
Sky_hidden=tf.nn.relu(tf.matmul(Sky_input,Sky_weights)+Sky_bias)




Semantic_weights=tf.Variable(tf.random_normal(shape=[5*hidden_size,hidden_size],
                             stddev=0.01),name="Semantic_weights")
Semantic_bias=tf.Variable(tf.zeros([1,hidden_size]),name="Semantic_bias")
Semantic_shared=tf.matmul(tf.concat([Ground_hidden,Objects_hidden,Building_hidden,Vegetation_hidden,Sky_hidden],1)
                                     ,Semantic_weights)+Semantic_bias



Semantic_deweights=tf.Variable(tf.random_normal(shape=[hidden_size,5*hidden_size],stddev=0.01),name="Semantic_deweights")
Semantic_debias=tf.Variable(tf.zeros([1,5*hidden_size]),name="Semantic_debias")  
decoder_sem=tf.matmul(Semantic_shared,Semantic_deweights)+Semantic_debias

decoder_ground,decoder_objects,decoder_building,decoder_vegetation,decoder_sky=tf.split(decoder_sem,num_or_size_splits=5, axis=1)



Ground_outweights=tf.Variable(pre_Ground_outweights,name="Ground_outweights")
Ground_outbias=tf.Variable(pre_Ground_outbias,name="Ground_outbias")
Ground_out=tf.nn.sigmoid(tf.matmul(decoder_ground,Ground_outweights)+Ground_outbias)


Objects_outweights=tf.Variable(pre_Objects_outweights,name="Objects_outweights")
Objects_outbias=tf.Variable(pre_Objects_outbias,name="Objects_outbias")
Objects_out=tf.nn.sigmoid(tf.matmul(decoder_objects,Objects_outweights)+Objects_outbias)



Building_outweights=tf.Variable(pre_Building_outweights,name="Building_outweights")
Building_outbias=tf.Variable(pre_Building_outbias,name="Building_outbias")
Building_out=tf.nn.sigmoid(tf.matmul(decoder_building,Building_outweights)+Building_outbias)


Vegetation_outweights=tf.Variable(pre_Vegetation_outweights,name="Vegetation_outweights")
Vegetation_outbias=tf.Variable(pre_Vegetation_outbias,name="Vegetation_outbias")
Vegetation_out=tf.nn.sigmoid(tf.matmul(decoder_vegetation,Vegetation_outweights)+Vegetation_outbias)


Sky_outweights=tf.Variable(pre_Sky_outweights,name="Sky_outweights")
Sky_outbias=tf.Variable(pre_Sky_outbias,name="Sky_outbias")
Sky_out=tf.nn.sigmoid(tf.matmul(decoder_sky,Sky_outweights)+Sky_outbias)



loss_sem=(tf.nn.l2_loss(Ground_input-Ground_out)+tf.nn.l2_loss(Objects_input-Objects_out)+
         tf.nn.l2_loss(Building_input-Building_out)+tf.nn.l2_loss(Vegetation_input-Vegetation_out)
         +tf.nn.l2_loss(Sky_input-Sky_out))



regularization_sem=(tf.nn.l2_loss(Ground_weights)
               +tf.nn.l2_loss(Objects_weights)
               +tf.nn.l2_loss(Building_weights)
               +tf.nn.l2_loss(Vegetation_weights)
               +tf.nn.l2_loss(Sky_weights)
               +tf.nn.l2_loss(Semantic_weights)
               +tf.nn.l2_loss(Semantic_deweights)
               +tf.nn.l2_loss(Ground_outweights)
               +tf.nn.l2_loss(Objects_outweights)
               +tf.nn.l2_loss(Building_outweights)
               +tf.nn.l2_loss(Vegetation_outweights)
               +tf.nn.l2_loss(Sky_outweights)
               )

loss_sem=loss_sem+1e-5*regularization_sem

first_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                 "Sem")


optimizer_sem=tf.train.AdamOptimizer(learning_rate=1e-5,
                                      beta1=0.9,beta2=0.999,
                                      epsilon=1e-3,use_locking=False,
                                      name='Adam').minimize(loss_sem,var_list=first_train_vars)

optimizer_sem_full=tf.train.AdamOptimizer(learning_rate=1e-7,
                                      beta1=0.9,beta2=0.999,
                                      epsilon=1e-3,use_locking=False,
                                      name='Adam').minimize(loss_sem)





par_path="../par/"
if not os.path.isdir(par_path):
    os.mkdir(par_path)

model_path="../model"
if not os.path.isdir(model_path):
    os.mkdir(model_path)



init=tf.global_variables_initializer()
saver=tf.train.Saver()
train_size=Ground_data.shape[0]
train_indices=range(train_size)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5

with tf.Session(config=config) as sess:
    sess.run(init)
    for i in range(30):
        
        perm_indices=np.random.permutation(train_indices)
        
        for step in range(int(train_size/batch_size)):
            
            offset=(step*batch_size)%(train_size-batch_size)
            batch_indices=perm_indices[offset:(offset+batch_size)]
            
            
            feed_dict={Ground_input:Ground_data[batch_indices,:],
                       Objects_input:Objects_data[batch_indices,:],
                       Vegetation_input:Vegetation_data[batch_indices,:],
                       Building_input:Building_data[batch_indices,:],
                       Sky_input:Sky_data[batch_indices,:]
                      }
            
            _sem,l_sem=sess.run([optimizer_sem,loss_sem],feed_dict=feed_dict)
    
        print("loss of semantic is :",l_sem) 

    for i in range(30,num_epoch):
        
        perm_indices=np.random.permutation(train_indices)
        
        for step in range(int(train_size/batch_size)):
            
            offset=(step*batch_size)%(train_size-batch_size)
            batch_indices=perm_indices[offset:(offset+batch_size)]
            
            
            feed_dict={Ground_input:Ground_data[batch_indices,:],
                       Objects_input:Objects_data[batch_indices,:],
                       Vegetation_input:Vegetation_data[batch_indices,:],
                       Building_input:Building_data[batch_indices,:],
                       Sky_input:Sky_data[batch_indices,:]
                      }
            
            _sem,l_sem=sess.run([optimizer_sem_full,loss_sem],feed_dict=feed_dict)

        print("loss of semantic is :",l_sem)      


    saver.save(sess,model_path+"/pretraining_sem_full.ckpt")
            
            
    print("training shared sem finished !")

    np.save(par_path+"Ground_weights",sess.run(Ground_weights))
    np.save(par_path+"Ground_bias",sess.run(Ground_bias))
    np.save(par_path+"Ground_outweights",sess.run(Ground_outweights))
    np.save(par_path+"Ground_outbias",sess.run(Ground_outbias))
    
    np.save(par_path+"Objects_weights",sess.run(Objects_weights))
    np.save(par_path+"Objects_bias",sess.run(Objects_bias))
    np.save(par_path+"Objects_outweights",sess.run(Objects_outweights))
    np.save(par_path+"Objects_outbias",sess.run(Objects_outbias))
    
    np.save(par_path+"Building_weights",sess.run(Building_weights))
    np.save(par_path+"Building_bias",sess.run(Building_bias))
    np.save(par_path+"Building_outweights",sess.run(Building_outweights))
    np.save(par_path+"Building_outbias",sess.run(Building_outbias))
    
    np.save(par_path+"Vegetation_weights",sess.run(Vegetation_weights))
    np.save(par_path+"Vegetation_bias",sess.run(Vegetation_bias))
    np.save(par_path+"Vegetation_outweights",sess.run(Vegetation_outweights))        
    np.save(par_path+"Vegetation_outbias",sess.run(Vegetation_outbias)) 
    
    np.save(par_path+"Sky_weights",sess.run(Sky_weights))
    np.save(par_path+"Sky_bias",sess.run(Sky_bias))
    np.save(par_path+"Sky_outweights",sess.run(Sky_outweights))
    np.save(par_path+"Sky_outbias",sess.run(Sky_outbias)) 
            

    np.save(par_path+"Semantic_deweights",sess.run(Semantic_deweights))
    np.save(par_path+"Semantic_debias",sess.run(Semantic_debias))
    np.save(par_path+"Semantic_weights",sess.run(Semantic_weights))
    np.save(par_path+"Semantic_bias",sess.run(Semantic_bias))


