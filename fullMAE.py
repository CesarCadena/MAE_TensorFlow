
# coding: utf-8

# In[3]:

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

batch_size=128
num_epochs=1
learning_rate=1e-3
hidden_size=1024

# ##  load data

# In[ ]:

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


# In[10]:

#load parameters

pre_Depth_weights=np.load("../par/Depth_weights.npy")
pre_Depth_bias=np.load("../par/Depth_bias.npy")
pre_Depth_outweights=np.load("../par/Depth_outweights.npy")
pre_Depth_outbias=np.load("../par/Depth_outbias.npy")

pre_Red_weights=np.load("../par/Red_weights.npy")
pre_Red_bias=np.load("../par/Red_bias.npy")
pre_Red_outweights=np.load("../par/Red_outweights.npy")
pre_Red_outbias=np.load("../par/Red_outbias.npy")

pre_Green_weights=np.load("../par/Green_weights.npy")
pre_Green_bias=np.load("../par/Green_bias.npy")
pre_Green_outweights=np.load("../par/Green_outweights.npy")
pre_Green_outbias=np.load("../par/Green_outbias.npy")

pre_Blue_weights=np.load("../par/Blue_weights.npy")
pre_Blue_bias=np.load("../par/Blue_bias.npy")
pre_Blue_outweights=np.load("../par/Blue_outweights.npy")
pre_Blue_outbias=np.load("../par/blue_outbias.npy")

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


pre_Sem_en_weights=np.load("../par/Sem_en_weights.npy")
pre_Sem_en_bias=np.load("../par/Sem_en_bias.npy")
pre_Sem_de_weights=np.load("../par/Sem_de_weights.npy")
pre_Sem_de_bias=np.load("../par/Sem_de_bias.npy")





# ##  full MAE

# In[11]:

### build full MAE  model 
# totally 9 channels 
Red_input=tf.placeholder(tf.float32,shape=[batch_size,1080])
Blue_input=tf.placeholder(tf.float32,shape=[batch_size,1080])
Green_input=tf.placeholder(tf.float32,shape=[batch_size,1080])
Depth_input=tf.placeholder(tf.float32,shape=[batch_size,1080])
Depthmask_input=tf.placeholder(tf.float32,shape=[batch_size,1080])

Ground_input=tf.placeholder(tf.float32,shape=[batch_size,1080])
Objects_input=tf.placeholder(tf.float32,shape=[batch_size,1080])
Building_input=tf.placeholder(tf.float32,shape=[batch_size,1080])
Vegetation_input=tf.placeholder(tf.float32,shape=[batch_size,1080])
Sky_input=tf.placeholder(tf.float32,shape=[batch_size,1080])



Depth_weights=tf.Variable(pre_Depth_weights,name="Depth_weights")
Depth_bias=tf.Variable(pre_Depth_bias,name="Depth_bias")


Red_weights=tf.Variable(pre_Red_weights,name="Red_weights")
Red_bias=tf.Variable(pre_Red_bias,name="Red_bias")


Blue_weights=tf.Variable(pre_Blue_weights,name="Blue_weights")
Blue_bias=tf.Variable(pre_Blue_bias,name="Blue_bias")


Green_weights=tf.Variable(pre_Green_weights,name="Green_weights")
Green_bias=tf.Variable(pre_Green_bias,name="Green_bias")


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


Depth_hidden=tf.nn.relu(tf.matmul(Depth_input,Depth_weights)+Depth_bias)
Red_hidden=tf.nn.relu(tf.matmul(Red_input,Red_weights)+Red_bias)
Blue_hidden=tf.nn.relu(tf.matmul(Blue_input,Blue_weights)+Blue_bias)
Green_hidden=tf.nn.relu(tf.matmul(Green_input,Green_weights)+Green_bias)
Ground_hidden=tf.nn.relu(tf.matmul(Ground_input,Ground_weights)+Ground_bias)
Objects_hidden=tf.nn.relu(tf.matmul(Objects_input,Objects_weights)+Objects_bias)
Building_hidden=tf.nn.relu(tf.matmul(Building_input,Building_weights)+Building_bias)
Vegetation_hidden=tf.nn.relu(tf.matmul(Vegetation_input,Vegetation_weights)+Vegetation_bias)
Sky_hidden=tf.nn.relu(tf.matmul(Sky_input,Sky_weights)+Sky_bias)


Semantic_weights=tf.Variable(pre_Sem_en_weights,name="Semantic_weights")
Semantic_bias=tf.Variable(pre_Sem_en_bias,name="Semantic_bias")

Semantic_shared=tf.matmul(tf.concat([Ground_hidden,Objects_hidden,Building_hidden,Vegetation_hidden,Sky_hidden],1)
                                     ,Semantic_weights)+Semantic_bias



Fullshared_weights=tf.Variable(tf.random_normal(shape=[5*hidden_size,hidden_size],
                               stddev=0.01),name="Fullshared_weights")
Fullshared_bias=tf.Variable(tf.zeros([1,hidden_size]),name="Fullshared_bias")

Full_shared=tf.matmul(tf.concat([Depth_hidden,Red_hidden,Green_hidden,Blue_hidden,
                                            Semantic_shared],1),Fullshared_weights)+Fullshared_bias



#####decoder 

decoder_weights=tf.Variable(tf.random_normal(shape=[hidden_size,5*hidden_size],
                           stddev=0.01),name="decoder_weights")
decoder_bias=tf.Variable(tf.zeros([1,5*hidden_size]),name="decoder_bias")

decoder_layer=tf.matmul(Full_shared,decoder_weights)+decoder_bias


decoder_Depth,decoder_Red,decoder_Green,decoder_Blue,decoder_sem=tf.split(decoder_layer,num_or_size_splits=5, axis=1)


decoder_semweights=tf.Variable(pre_Sem_de_weights,name="decoder_semweights")
decoder_sembias=tf.Variable(pre_Sem_de_bias,name="decoder_sembias")
decoder_sems=tf.matmul(decoder_sem,decoder_semweights)+decoder_sembias


decoder_ground,decoder_objects,decoder_building,decoder_vegetation,decoder_sky=tf.split(decoder_sems,num_or_size_splits=5, axis=1)



Depth_outweights=tf.Variable(pre_Depth_outweights,name="Depth_outweights")
Depth_outbias=tf.Variable(pre_Depth_outbias,name="Depth_outbias")
Depth_out=tf.matmul(decoder_Depth,Depth_outweights)+Depth_outbias


Red_outweights=tf.Variable(pre_Red_outweights,name="Red_outweights")
Red_outbias=tf.Variable(pre_Red_outbias,name="Red_outbias")
Red_out=tf.nn.sigmoid(tf.matmul(decoder_Red,Red_outweights)+Red_outbias)

Green_outweights=tf.Variable(pre_Green_outweights,name="Green_outweights")
Green_outbias=tf.Variable(pre_Green_outbias,name="Green_outbias")
Green_out=tf.nn.sigmoid(tf.matmul(decoder_Green,Green_outweights)+Green_outbias)

Blue_outweights=tf.Variable(pre_Blue_outweights,name="Blue_outweights")
Blue_outbias=tf.Variable(pre_Blue_outbias,name="Blue_outbias")
Blue_out=tf.nn.sigmoid(tf.matmul(decoder_Blue,Blue_outweights)+Blue_outbias)


Ground_outweights=tf.Variable(pre_Ground_outweights,name="ground_outweights")
Ground_outbias=tf.Variable(pre_Ground_outbias,name="ground_outbias")
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




# be careful with this loss function

loss=(tf.nn.l2_loss(Red_input-Red_out)
     +tf.nn.l2_loss(Blue_input-Blue_out)
     +tf.nn.l2_loss(Green_input-Green_out)
     +tf.nn.l2_loss(np.multiply((Depth_input-Depth_out),Depthmask_input))
     +tf.nn.l2_loss(Ground_input-Ground_out)
     +tf.nn.l2_loss(Objects_input-Objects_out)
     +tf.nn.l2_loss(Building_input-Building_out)
     +tf.nn.l2_loss(Vegetation_input-Vegetation_out)
     +tf.nn.l2_loss(Sky_input-Sky_out))



regularization=(tf.nn.l2_loss(Depth_weights)+tf.nn.l2_loss(Depth_outbias)
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

              
loss_r=loss+1e-8*regularization

optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=0.9,beta2=0.999,
                                 epsilon=0.001,use_locking=False,name='Adam').minimize(loss_r)
#optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)


# ##  training

# In[41]:

saver=tf.train.Saver()

init=tf.global_variables_initializer()
train_size=Ground_data.shape[0]
train_indices=range(train_size)


with tf.Session() as sess:
    

    sess.run(init)

    #summary_op=tf.summary.merge_all()
    #summary_writer=tf.summary.FileWriter(FLAGS.train_dir,graph=sess.graph)

    for ipoch in range(num_epochs):
        

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
                       Depthmask_input:Depthmask_data[batch_indices,:]
                      }   

      
            _,l=sess.run([optimizer,loss_r],feed_dict=feed_dict)
            print ('loss is ' ,l)
    
    save_path=saver.save(sess,'../model/MAE.ckpt')

        
    print("final loss is :%d\n." % l)
    print("Model saved in file: %s" % save_path)





