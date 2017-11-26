import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from process_data import  process_data

from  CAE_rgb  import  input_hidden_rgb,hidden_output_rgb,sem_loss
from  CAE_sem  import  input_hidden_sem,hidden_output_sem
from  CAE_depth  import  input_hidden_depth,hidden_output_depth

#import CAE_sem.py as CAE_sem
#import CAE_depth.py as CAE_depth

batch_size=20
num_epochs=100
hidden_size=1024
RESTORE=0
SEED = None

# process  data  
data=process_data('training')
#RGB data
Red_data=data['Red']
Green_data=data['Green']
Blue_data=data['Blue']
Red_data=np.reshape(Red_data,[-1,60,18,1])
Blue_data=np.reshape(Blue_data,[-1,60,18,1])
Green_data=np.reshape(Green_data,[-1,60,18,1])
RGB_data=np.concatenate([Red_data,Blue_data,Green_data],axis=3)
#RGB_datashape:(num,60,18,3)

#semantic data 
Ground_data=data['Ground']
Objects_data=data['Objects']
Building_data=data['Building']
Vegetation_data=data['Vegetation']
Sky_data=data['Sky']
Ground_data=np.reshape(Ground_data,[-1,60,18,1])
Objects_data=np.reshape(Objects_data,[-1,60,18,1])
Building_data=np.reshape(Building_data,[-1,60,18,1])
Vegetation_data=np.reshape(Vegetation_data,[-1,60,18,1])
Sky_data=np.reshape(Sky_data,[-1,60,18,1])
Sem_data=np.concatenate([Ground_data,Objects_data,Building_data,Vegetation_data,Sky_data],axis=3)
#Sem_data shape :(num,60,18,5)

# depth data 
Depth_data=data['Depth']
Depthmask_data=data['Depthmask']
Depth_data=np.multiply(Depth_data,Depthmask_data)
Depth_data=np.reshape(Depth_data,[-1,60,18,1])
#  depth data shape:(num,60,18,1)

print(RGB_data.shape)
print(Sem_data.shape)
print(Depth_data.shape)
def weight_variable(name,shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.get_variable(name=name,dtype=tf.float32,
                           initializer=initial,trainable=True)
def bias_variable(name,shape):
    initial=0.1+tf.zeros(shape)
    return tf.get_variable(name=name,dtype=tf.float32,
                          initializer=initial,trainable=True)



#### build the model
with tf.name_scope("sem_input"):
        x_sem= tf.placeholder(tf.float32, shape=[None,60,18,5])
with tf.name_scope("depth_input"):
        x_depth= tf.placeholder(tf.float32, shape=[None,60,18,1]) 
with tf.name_scope("rgb_input"):
        x_rgb= tf.placeholder(tf.float32, shape=[None,60,18,3])

#hidden layers 1024
h_fc_sem=input_hidden_sem(x_sem)
h_fc_depth=input_hidden_depth(x_depth)
h_fc_rgb=input_hidden_rgb(x_rgb)



with tf.name_scope('seperate_shared'):
    with tf.variable_scope("seperate_shared"):
        w=weight_variable(name='weights',shape=[3*1024,1024])
        b=bias_variable(name='bias',shape=[1024])

    All_input=tf.concat([h_fc_rgb,h_fc_sem,h_fc_depth],axis=1)
    fullshared=tf.nn.relu(tf.matmul(All_input,w)+b)


with tf.name_scope('shared_seperate'):
    with tf.variable_scope("shared_seperate"):
        w=weight_variable(name='weights',shape=[1024,3*1024])
        b=bias_variable(name='bias',shape=[3*1024])
    All_out=tf.matmul(fullshared,w)+b

fc_out_rgb,fc_out_sem,fc_out_depth=tf.split(All_out,[1024,1024,1024],1)


rgb_out=hidden_output_rgb(fc_out_rgb)
sem_out=hidden_output_sem(fc_out_sem)
depth_out=hidden_output_depth(fc_out_depth)


with tf.name_scope("loss"):
    loss=(tf.reduce_sum(tf.nn.l2_loss(rgb_out-x_rgb))
         +tf.reduce_sum(tf.nn.l2_loss(depth_out-x_depth))
         +sem_loss(x_sem,sem_out))



varlist=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope=['seperate_shared','shared_seperate'])


with tf.name_scope("depth_train"):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss,var_list=varlist)



train_size=RGB_data.shape[0]
train_indices=range(train_size)
init=tf.global_variables_initializer()

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction =0.5


tf.summary.scalar('loss',loss)
summary_op=tf.summary.merge_all()


with tf.Session(config=config) as sess:

    sess.run(init)
    writer=tf.summary.FileWriter('../graphs',sess.graph)

    for ipoch in range(1):
        perm_indices=np.random.permutation(train_indices)

        for step in range(int(train_size/batch_size)):

            offset=(step*batch_size)%(train_size-batch_size)
            batch_indices=perm_indices[offset:(offset+batch_size)]
            
            _,summary=sess.run([train_step,summary_op],feed_dict={x:data[batch_indices],keep_prob:0.5})

            writer.add_summary(summary,step)

        print('ipoch:',ipoch)