# coding: utf-8
# %load autoencoder_tensorflow_MAE.py
# Draft MAE Model in tensorflow based on Dr.Cesar Cadena
# this code is developed by Yi Liu 
#run the code under the folder ~/project
#this version has converged!

import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from process_data import  process_data

tf.reset_default_graph()


batch_size=20
num_epochs=100
hidden_size=1024
RESTORE=0
SEED = None


print("###########loading data.........##########")
#  prepare data 
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
Depth_data=np.multiply(Depth_data,Depthmask_data)



# data augmentation with lack of depth and semantic channels

Red_data=np.concatenate((Red_data,Red_data),axis=0)
Blue_data=np.concatenate((Blue_data,Blue_data),axis=0)
Green_data=np.concatenate((Green_data,Green_data),axis=0)

Depth_data=np.concatenate((Depth_data,Depth_data*0),axis=0)
Depth_data_label=np.concatenate((Depth_data,Depth_data),axis=0)

Ground_data=np.concatenate((Ground_data,Ground_data*0),axis=0)
Ground_data_label=np.concatenate((Ground_data,Ground_data),axis=0)

Objects_data=np.concatenate((Objects_data,Objects_data*0),axis=0)
Objects_data_label=np.concatenate((Objects_data,Objects_data),axis=0)

Building_data=np.concatenate((Building_data,Building_data*0),axis=0)
Building_data_label=np.concatenate((Building_data,Building_data),axis=0)

Vegetation_data=np.concatenate((Vegetation_data,Vegetation_data*0),axis=0)
Vegetation_data_label=np.concatenate((Vegetation_data,Vegetation_data),axis=0)

Sky_data=np.concatenate((Sky_data,Sky_data*0),axis=0)
Sky_data_label=np.concatenate((Sky_data,Sky_data),axis=0)



print("##########building model ........#######")
### build full MAE  model 
# totally 9 channels (including mask channels)
Red_input=tf.placeholder(tf.float32,shape=[None,1080])
Blue_input=tf.placeholder(tf.float32,shape=[None,1080])
Green_input=tf.placeholder(tf.float32,shape=[None,1080])
Depth_input=tf.placeholder(tf.float32,shape=[None,1080])
Ground_input=tf.placeholder(tf.float32,shape=[None,1080])
Objects_input=tf.placeholder(tf.float32,shape=[None,1080])
Building_input=tf.placeholder(tf.float32,shape=[None,1080])
Vegetation_input=tf.placeholder(tf.float32,shape=[None,1080])
Sky_input=tf.placeholder(tf.float32,shape=[None,1080])
#auxiliary channel

Depth_label=tf.placeholder(tf.float32,shape=[None,1080])
Ground_label=tf.placeholder(tf.float32,shape=[None,1080])
Objects_label=tf.placeholder(tf.float32,shape=[None,1080])
Building_label=tf.placeholder(tf.float32,shape=[None,1080])
Vegetation_label=tf.placeholder(tf.float32,shape=[None,1080])
Sky_label=tf.placeholder(tf.float32,shape=[None,1080])
#Depthmask_input=tf.placeholder(tf.float32,shape=[None,1080])



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

Semantic_shared=tf.matmul(tf.concat([Ground_hidden,Objects_hidden,Building_hidden,Vegetation_hidden,Sky_hidden],1)
                                     ,Semantic_weights)+Semantic_bias




Fullshared_weights=tf.Variable(tf.random_normal(shape=[5*hidden_size,hidden_size],
                               stddev=0.1),name="Fullshared_weights")
Fullshared_bias=tf.Variable(tf.zeros([1,hidden_size]),name="Fullshared_bias")

Full_shared=tf.matmul(tf.concat([Depth_hidden,Red_hidden,Green_hidden,Blue_hidden,
                                            Semantic_shared],1),Fullshared_weights)+Fullshared_bias





#####decoder######

Fullshared_deweights=tf.Variable(tf.random_normal(shape=[hidden_size,5*hidden_size],
                           stddev=0.1),name="Fullshared_deweights")
Fullshared_debias=tf.Variable(tf.zeros([1,5*hidden_size]),name="Fullshared_debias")

decoder_layer=tf.matmul(Full_shared,Fullshared_deweights)+Fullshared_debias



decoder_Depth,decoder_Red,decoder_Green,decoder_Blue,decoder_sem=tf.split(decoder_layer,num_or_size_splits=5, axis=1)



Semantic_deweights=tf.Variable(tf.random_normal(shape=[hidden_size,5*hidden_size],
                                           stddev=0.01),name="Semantic_deweights")
Semantic_debias=tf.Variable(tf.zeros([1,5*hidden_size]),name="Semantic_debias")
decoder_sems=tf.matmul(decoder_sem,Semantic_deweights)+Semantic_debias




decoder_Ground,decoder_Objects,decoder_Building,decoder_Vegetation,decoder_Sky=tf.split(decoder_sems,num_or_size_splits=5, axis=1)


Depth_outweights=tf.Variable(tf.random_normal(shape=[hidden_size,1080],
                                   stddev=0.01),name="Depth_outweights")
Depth_outbias=tf.Variable(tf.zeros([1,1080]),name="Depth_outbias")
Depth_out=tf.matmul(decoder_Depth,Depth_outweights)+Depth_outbias



Red_outweights=tf.Variable(tf.random_normal(shape=[hidden_size,1080],
                                   stddev=0.01),name="Red_outweights")
Red_outbias=tf.Variable(tf.zeros([1,1080]),name="Red_outbias")
Red_out=tf.matmul(decoder_Red,Red_outweights)+Red_outbias

Green_outweights=tf.Variable(tf.random_normal(shape=[hidden_size,1080],
                                   stddev=0.01),name="Green_outweights")
Green_outbias=tf.Variable(tf.zeros([1,1080]),name="Green_outbias")
Green_out=tf.matmul(decoder_Green,Green_outweights)+Green_outbias

Blue_outweights=tf.Variable(tf.random_normal(shape=[hidden_size,1080],
                                   stddev=0.01),name="Blue_outweights")
Blue_outbias=tf.Variable(tf.zeros([1,1080]),name="Blue_outbias")
Blue_out=tf.matmul(decoder_Blue,Blue_outweights)+Blue_outbias


Ground_outweights=tf.Variable(tf.random_normal(shape=[hidden_size,1080],
                                   stddev=0.01),name="ground_outweights")
Ground_outbias=tf.Variable(tf.zeros([1,1080]),name="ground_outbias")
Ground_out=tf.nn.sigmoid(tf.matmul(decoder_Ground,Ground_outweights)+Ground_outbias)

Objects_outweights=tf.Variable(tf.random_normal(shape=[hidden_size,1080],
                                   stddev=0.01),name="Objects_outweights")
Objects_outbias=tf.Variable(tf.zeros([1,1080]),name="Objects_outbias")
Objects_out=tf.nn.sigmoid(tf.matmul(decoder_Objects,Objects_outweights)+Objects_outbias)


Building_outweights=tf.Variable(tf.random_normal(shape=[hidden_size,1080],
                                   stddev=0.01),name="Building_outweights")
Building_outbias=tf.Variable(tf.zeros([1,1080]),name="Building_outbias")
Building_out=tf.nn.sigmoid(tf.matmul(decoder_Building,Building_outweights)+Building_outbias)


Vegetation_outweights=tf.Variable(tf.random_normal(shape=[hidden_size,1080],
                                   stddev=0.01),name="Vegetation_outweights")
Vegetation_outbias=tf.Variable(tf.zeros([1,1080]),name="Vegetation_outbias")
Vegetation_out=tf.nn.sigmoid(tf.matmul(decoder_Vegetation,Vegetation_outweights)+Vegetation_outbias)

Sky_outweights=tf.Variable(tf.random_normal(shape=[hidden_size,1080],
                                   stddev=0.01),name="Sky_outweights")
Sky_outbias=tf.Variable(tf.zeros([1,1080]),name="Sky_outbias")
Sky_out=tf.nn.sigmoid(tf.matmul(decoder_Sky,Sky_outweights)+Sky_outbias)


# be careful with this loss function


loss=(tf.nn.l2_loss(Red_input-Red_out)
     +tf.nn.l2_loss(Blue_input-Blue_out)
     +tf.nn.l2_loss(Green_input-Green_out)
     +100*tf.nn.l2_loss(Depth_label-Depth_out)
     -tf.reduce_sum(np.multiply(Ground_label,tf.log(tf.clip_by_value(Ground_out,1e-10,1)))
                          +np.multiply(1-Ground_label,tf.log(tf.clip_by_value(1-Ground_out,1e-10,1)))
                          )-tf.reduce_sum(np.multiply(Objects_label,tf.log(tf.clip_by_value(Objects_out,1e-10,1)))
                          +np.multiply(1-Objects_label,tf.log(tf.clip_by_value(1-Objects_out,1e-10,1)))
                          )-tf.reduce_sum(np.multiply(Building_label,tf.log(tf.clip_by_value(Building_out,1e-10,1)))
                          +np.multiply(1-Building_label,tf.log(tf.clip_by_value(1-Building_out,1e-10,1)))
                          )-tf.reduce_sum(np.multiply(Sky_label,tf.log(tf.clip_by_value(Sky_out,1e-10,1)))
                          +np.multiply(1-Sky_label,tf.log(tf.clip_by_value(1-Sky_out,1e-10,1)))
                          )-tf.reduce_sum(np.multiply(Vegetation_label,tf.log(tf.clip_by_value(Vegetation_out,1e-10,1)))
                          +np.multiply(1-Vegetation_label,tf.log(tf.clip_by_value(1-Vegetation_out,1e-10,1)))
                          ))


regularizer = tf.contrib.layers.l2_regularizer(scale=5e-04)
regularization= tf.contrib.layers.apply_regularization(regularizer,
                       weights_list=[Depth_weights,Red_weights,Blue_weights,
                                     Green_weights,Ground_weights,Objects_weights,
                                     Building_weights,Vegetation_weights,Sky_weights,
                                     Semantic_weights,Fullshared_weights,Fullshared_deweights,
                                     Depth_outweights,Red_outweights,Blue_outweights,
                                     Green_outweights,Ground_outweights,Objects_outweights,
                                     Building_outweights,Vegetation_outweights,Sky_outweights])
        

              
loss_r=loss+regularization

first_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                 "Full")

optimizer_first=tf.train.AdamOptimizer(learning_rate=1e-4,beta1=0.9,beta2=0.999,
                
                                 epsilon=1e-8,use_locking=False,name='Adam').minimize(loss_r,var_list=first_train_vars)



optimizer_second=tf.train.AdamOptimizer(learning_rate=1e-6,beta1=0.9,beta2=0.999,
                                 epsilon=1e-8,use_locking=False,name='Adam').minimize(loss_r)



model_path="../model_full"
if not os.path.isdir(model_path):
    os.mkdir(model_path)

Semantic_list={"Ground_weights": Ground_weights,
              "Ground_bias":Ground_bias,
              "Building_weights":Building_weights,
              "Building_bias":Building_bias,
              "Objects_weights":Objects_weights,
              "Objects_bias":Objects_bias,
              "Sky_weights":Sky_weights,
              "Sky_bias":Sky_bias,
              "Vegetation_weights":Vegetation_weights,
              "Vegetation_bias":Vegetation_bias,
              "Ground_outweights":Ground_outweights,
              "Ground_outbias":Ground_outbias,
              "Building_outweights":Building_outweights,
              "Building_outbias":Building_outbias,
              "Objects_outweights":Objects_outweights,
              "Objects_outbias":Objects_outbias,
              "Sky_outweights":Sky_outweights,
              "Sky_outbias":Sky_outbias,
              "Vegetation_outweights":Vegetation_outweights,
              "Vegetation_outbias":Vegetation_outbias,
              "Semantic_weights":Semantic_weights,
              "Semantic_bias":Semantic_bias,
              "Semantic_deweights":Semantic_deweights,
              "Semantic_debias":Semantic_debias}

Sep_list={"Red_weights":Red_weights,
          "Red_bias":Red_bias,
          "Red_outweights":Red_outweights,
          "Red_outbias":Red_outbias,
          "Blue_weights":Blue_weights,
          "Blue_bias":Blue_bias,
          "Blue_outweights":Blue_outweights,
          "Blue_outbias":Blue_outbias,
          "Green_weights":Green_weights,
          "Green_bias":Green_bias,
          "Green_outweights":Green_outweights,
          "Green_outbias":Green_outbias,
          "Depth_weights":Depth_weights,
          "Depth_bias":Depth_bias,
          "Depth_outweights":Depth_outweights,
          "Depth_outbias":Depth_outbias}

####  training
saver_sep=tf.train.Saver(Sep_list)
saver_sem=tf.train.Saver(Semantic_list)

saver_full=tf.train.Saver()
init=tf.global_variables_initializer()


train_size=Ground_data.shape[0]
train_indices=range(train_size)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction =0.5


print("#########start training .......################")
with tf.Session(config=config) as sess:

  if RESTORE:

    saver_full.restore(sess,'../model_full/MAE.ckpt')

  else:

    sess.run(init)
    saver_sep.restore(sess,"../model/pretraining_sep.ckpt")
    saver_sem.restore(sess,"../model_sem/pretraining_sem_full.ckpt")


    #summary_op=tf.summary.merge_all()
    #summary_writer=tf.summary.FileWriter(FLAGS.train_dir,graph=sess.graph)

    for ipoch in range(500):
        
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
                       Depth_input:Depth_data[batch_indices,:],

                       Depth_label:Depth_data_label[batch_indices,:],
                       Sky_label:Sky_data_label[batch_indices,:],
                       Building_label:Building_data_label[batch_indices,:],
                       Vegetation_label:Vegetation_data_label[batch_indices,:],
                       Objects_label:Objects_data_label[batch_indices,:],
                       Ground_label:Ground_data_label[batch_indices,:],
                       #Depthmask_input:Depthmask_data[batch_indices,:]
                      }   
    
            _,l,r=sess.run([optimizer_first,loss,regularization],feed_dict=feed_dict)

        print ('loss of epoch %d :'%ipoch,l)
        print ('regularization:',r)




    for ipoch in range(500,num_epochs):
        
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
                       Depth_input:Depth_data[batch_indices,:],

                       Depth_label:Depth_data_label[batch_indices,:],
                       Sky_label:Sky_data_label[batch_indices,:],
                       Building_label:Building_data_label[batch_indices,:],
                       Vegetation_label:Vegetation_data_label[batch_indices,:],
                       Objects_label:Objects_data_label[batch_indices,:],
                       Ground_label:Ground_data_label[batch_indices,:],
                       #Depthmask_input:Depthmask_data[batch_indices,:]
                      }   
    
            _,l,r=sess.run([optimizer_second,loss,regularization],feed_dict=feed_dict)


        print ('loss of epoch %d :'%ipoch,l)
        print ('regularization:',r)

    save_path=saver_full.save(sess,model_path+'/MAE.ckpt')

    print("final loss is :%d\n." % l)
    print("Model saved in file: %s" % save_path)
