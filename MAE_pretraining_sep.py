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
    


#RGB Depth model:

Red_input=tf.placeholder(tf.float32,shape=[None,1080])

Red_weights=tf.Variable(tf.random_normal(shape=[1080,hidden_size],
                                  stddev=0.01),name="Red_weights")
Red_bias=tf.Variable(tf.zeros([1,hidden_size]),name="Red_bias")

Red_hidden=tf.nn.relu(tf.matmul(Red_input,Red_weights)+Red_bias)

Red_outweights=tf.Variable(tf.random_normal(shape=[hidden_size,1080],stddev=0.01),name="Red_outweights")
Red_outbias=tf.Variable(tf.zeros([1,1080]),name="Red_outbias")
Red_out=tf.nn.sigmoid(tf.matmul(Red_hidden,Red_outweights)+Red_outbias)


loss_red=tf.nn.l2_loss(Red_input-Red_out)

regularization_red=(tf.nn.l2_loss(Red_weights)
               +tf.nn.l2_loss(Red_outweights))

loss_red_t=loss_red+1e-5*regularization_red

optimizer_red=tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss_red_t)

# Green model :



Green_input=tf.placeholder(tf.float32,shape=[None,1080])

Green_weights=tf.Variable(tf.random_normal(shape=[1080,hidden_size],
                                  stddev=0.01),name="Green_weights")
Green_bias=tf.Variable(tf.zeros([1,hidden_size]),name="Green_bias")

Green_hidden=tf.nn.relu(tf.matmul(Green_input,Green_weights)+Green_bias)

Green_outweights=tf.Variable(tf.random_normal(shape=[hidden_size,1080],stddev=0.01),name="Green_outweights")
Green_outbias=tf.Variable(tf.zeros([1,1080]),name="Green_outbias")
Green_out=tf.nn.sigmoid(tf.matmul(Green_hidden,Green_outweights)+Green_outbias)


loss_green=tf.nn.l2_loss(Green_input-Green_out)

regularization_green=(tf.nn.l2_loss(Green_weights)
                     +tf.nn.l2_loss(Green_outweights))

loss_green=loss_green+1e-5*regularization_green

optimizer_green=tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss_green)

# ## blue channel


Blue_input=tf.placeholder(tf.float32,shape=[None,1080])

Blue_weights=tf.Variable(tf.random_normal(shape=[1080,hidden_size],
                                  stddev=0.01),name="Blue_weights")

Blue_bias=tf.Variable(tf.zeros([1,hidden_size]),name="Blue_bias")

Blue_hidden=tf.nn.relu(tf.matmul(Blue_input,Blue_weights)+Blue_bias)

Blue_outweights=tf.Variable(tf.random_normal(shape=[hidden_size,1080],stddev=0.01),name="Blue_outweights")
Blue_outbias=tf.Variable(tf.zeros([1,1080]),name="Blue_outbias")
Blue_out=tf.nn.sigmoid(tf.matmul(Blue_hidden,Blue_outweights)+Blue_outbias)


loss_blue=tf.nn.l2_loss(Blue_input-Blue_out)

regularization_blue=(tf.nn.l2_loss(Blue_weights)
                     +tf.nn.l2_loss(Blue_outweights))

loss_blue=loss_blue+1e-5*regularization_blue

optimizer_blue=tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss_blue)



#### depth channel
Depth_input=tf.placeholder(tf.float32,shape=[None,1080])
Depthmask_input=tf.placeholder(tf.float32,shape=[None,1080])


Depth_weights=tf.Variable(tf.random_normal(shape=[1080,hidden_size],
                                  stddev=0.01),name="Depth_weights")


Depth_bias=tf.Variable(tf.zeros([1,hidden_size]),name="Depth_bias")
Depth_hidden=tf.nn.relu(tf.matmul(Depth_input,Depth_weights)+Depth_bias)


Depth_outweights=tf.Variable(tf.random_normal(shape=[hidden_size,1080],stddev=0.01),name="Depth_outweights")
Depth_outbias=tf.Variable(tf.zeros([1,1080]),name="Depth_outbias")
Depth_out=tf.matmul(Depth_hidden,Depth_outweights)+Depth_outbias




loss_depth=tf.nn.l2_loss(np.multiply(Depth_input,Depthmask_input)-np.multiply(Depth_out,Depthmask_input))

regularization_depth=(tf.nn.l2_loss(Depth_weights)
                     +tf.nn.l2_loss(Depth_outweights))

loss_depth_t=loss_depth+1e-4*regularization_depth

optimizer_depth=tf.train.AdamOptimizer(learning_rate=1e-5,
                                      beta1=0.9,beta2=0.999,
                                      epsilon=1e-8,use_locking=False,
                                      name='Adam').minimize(loss_depth_t)




######start Ground channel #######
Ground_input=tf.placeholder(tf.float32,shape=[None,1080])
Ground_weights=tf.Variable(tf.random_normal(shape=[1080,hidden_size],
                                   stddev=0.01),name="Ground_weights")
Ground_bias=tf.Variable(tf.zeros([1,hidden_size]),name="Ground_bias")
Ground_hidden=tf.nn.relu(tf.matmul(Ground_input,Ground_weights)+Ground_bias)

Ground_outweights=tf.Variable(tf.random_normal(shape=[hidden_size,1080],
                                   stddev=0.01),name="Ground_outweights")

Ground_outbias=tf.Variable(tf.zeros([1,1080]),name="Ground_outbias")
Ground_out=tf.nn.sigmoid(tf.matmul(Ground_hidden,Ground_outweights)+Ground_outbias)


loss_Ground=tf.nn.l2_loss(Ground_input-Ground_out)
regularization_Ground=(tf.nn.l2_loss(Ground_weights)
                      +tf.nn.l2_loss(Ground_outweights))
loss_Ground=loss_Ground+1e-5*regularization_Ground
optimizer_Ground=tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss_Ground)
###############

#Objects channel
Objects_input=tf.placeholder(tf.float32,shape=[None,1080])

Objects_weights=tf.Variable(tf.random_normal(shape=[1080,hidden_size],
                                   stddev=0.01),name="Objects_weights")

Objects_bias=tf.Variable(tf.zeros([1,hidden_size]),name="Objects_bias")

Objects_hidden=tf.nn.relu(tf.matmul(Objects_input,Objects_weights)+Objects_bias)

Objects_outweights=tf.Variable(tf.random_normal(shape=[hidden_size,1080],
                                    stddev=0.01),name="Objects_outweights")

Objects_outbias=tf.Variable(tf.zeros([1,1080]),name="Objects_outbias")

Objects_out=tf.nn.sigmoid(tf.matmul(Objects_hidden,Objects_outweights)+Objects_outbias)


loss_Objects=tf.nn.l2_loss(Objects_input-Objects_out)

regularization_Objects=(tf.nn.l2_loss(Objects_weights)+tf.nn.l2_loss(Objects_outweights))
             

loss_Objects=loss_Objects+1e-5*regularization_Objects

optimizer_Objects=tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss_Objects)
##########

###Building channels

Building_input=tf.placeholder(tf.float32,shape=[None,1080])

Building_weights=tf.Variable(tf.random_normal(shape=[1080,hidden_size],
                                   stddev=0.01),name="Building_weights")

Building_bias=tf.Variable(tf.zeros([1,hidden_size]),name="Building_bias")

Building_hidden=tf.nn.relu(tf.matmul(Building_input,Building_weights)+Building_bias)

Building_outweights=tf.Variable(tf.random_normal(shape=[hidden_size,1080],stddev=0.01)
                                   ,name="Building_outweights")

Building_outbias=tf.Variable(tf.zeros([1,1080]),name="Building_outbias")

Building_out=tf.nn.sigmoid(tf.matmul(Building_hidden,Building_outweights)+Building_outbias)

loss_Building=tf.nn.l2_loss(Building_input-Building_out)

regularization_Building=(tf.nn.l2_loss(Building_weights)
               +tf.nn.l2_loss(Building_outweights))

loss_Building=loss_Building+1e-5*regularization_Building

optimizer_Building=tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss_Building)

###########
#Sky channels

Sky_input=tf.placeholder(tf.float32,shape=[None,1080])

Sky_weights=tf.Variable(tf.random_normal(shape=[1080,hidden_size],
                                   stddev=0.01),name="Sky_weights")

Sky_bias=tf.Variable(tf.zeros([1,hidden_size]),name="Sky_bias")


Sky_hidden=tf.nn.relu(tf.matmul(Sky_input,Sky_weights)+Sky_bias)

Sky_outweights=tf.Variable(tf.random_normal(shape=[hidden_size,1080],stddev=0.01)
                                   ,name="Sky_outweights")
Sky_outbias=tf.Variable(tf.zeros([1,1080]),name="Sky_outbias")

Sky_out=tf.nn.sigmoid(tf.matmul(Sky_hidden,Sky_outweights)+Sky_outbias)

loss_Sky=tf.nn.l2_loss(Sky_input-Sky_out)
regularization_Sky=(tf.nn.l2_loss(Sky_weights)
               +tf.nn.l2_loss(Sky_outweights))

loss_Sky=loss_Sky+1e-5*regularization_Sky

optimizer_Sky=tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss_Sky)
#######
# channel Vegetation
                                                                                      
Vegetation_input=tf.placeholder(tf.float32,shape=[None,1080])

Vegetation_weights=tf.Variable(tf.random_normal(shape=[1080,hidden_size],
                                   stddev=0.01),name="Vegetation_weights")

Vegetation_bias=tf.Variable(tf.zeros([1,hidden_size]),name="Vegetation_bias")
Vegetation_hidden=tf.nn.relu(tf.matmul(Vegetation_input,Vegetation_weights)+Vegetation_bias)

Vegetation_outweights=tf.Variable(tf.random_normal(shape=[hidden_size,1080],stddev=0.01)
                                   ,name="Vegetation_outweights")

Vegetation_outbias=tf.Variable(tf.zeros([1,1080]),name="Vegetation_outbias")

Vegetation_out=tf.nn.sigmoid(tf.matmul(Vegetation_hidden,Vegetation_outweights)+Vegetation_outbias)
loss_Vegetation=tf.nn.l2_loss(Vegetation_input-Vegetation_out)

regularization_Vegetation=(tf.nn.l2_loss(Vegetation_weights)
               +tf.nn.l2_loss(Vegetation_outweights))

loss_Vegetation=loss_Vegetation+1e-5*regularization_Vegetation

optimizer_Vegetation=tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss_Vegetation)
#####

#Finish  Seperate Model  
#######


##start train seperate channels :

init=tf.global_variables_initializer()
saver=tf.train.Saver()
train_size=Red_data.shape[0]
train_indices=range(train_size)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5


par_path="../par/"
if not os.path.isdir(par_path):
    os.mkdir(par_path)

model_path="../model"
if not os.path.isdir(model_path):
    os.mkdir(model_path)



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
                       Depth_input:Depth_data[batch_indices,:],
                       Depthmask_input:Depthmask_data[batch_indices,:]
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

      
    saver.save(sess,model_path+"/pretraining_sep.ckpt")
     
    print ('seperately pretraining finished')

    np.save(par_path+"Red_weights",sess.run(Red_weights))
    np.save(par_path+"Red_bias",sess.run(Red_bias))
    np.save(par_path+"Red_outweights",sess.run(Red_outweights))
    np.save(par_path+"Red_outbias",sess.run(Red_outbias))
    
    np.save(par_path+"Green_weights",sess.run(Green_weights))
    np.save(par_path+"Green_bias",sess.run(Green_bias))
    np.save(par_path+"Green_outweights",sess.run(Green_outweights))
    np.save(par_path+"Green_outbias",sess.run(Green_outbias))
    
    np.save(par_path+"Blue_weights",sess.run(Blue_weights))
    np.save(par_path+"Blue_bias",sess.run(Blue_bias))
    np.save(par_path+"Blue_outweights",sess.run(Blue_outweights))
    np.save(par_path+"Blue_outbias",sess.run(Blue_outbias))
    

    np.save(par_path+"Depth_weights",sess.run(Depth_weights))
    np.save(par_path+"Depth_bias",sess.run(Depth_bias))
    np.save(par_path+"Depth_outweights",sess.run(Depth_outweights))
    np.save(par_path+"Depth_outbias",sess.run(Depth_outbias))


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

####################################
# start train the full shared MAE #
####################################

