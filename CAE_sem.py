import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from process_data import  process_data

batch_size=20
num_epochs=100
hidden_size=1024
RESTORE=0
SEED = None

#print("loading data.....")

"""
# preprocess data 
data=process_data('training')
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
data=np.concatenate([Ground_data,Objects_data,Building_data,Vegetation_data,Sky_data],axis=3)
print(data.shape)
#np.save('../sem(num,60,18,3)',data)
"""

data=np.load('../sem(num,60,18,5).npy')
print(data.shape)

def weight_variable(name,shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.get_variable(name=name,dtype=tf.float32,
                           initializer=initial,trainable=True)
def bias_variable(name,shape):
    initial=0.1+tf.zeros(shape)
    return tf.get_variable(name=name,dtype=tf.float32,
                          initializer=initial,trainable=True)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,3,3,1],padding='SAME')

def max_unpool_2x2(x, shape):
    inference =tf.image.resize_nearest_neighbor(x,tf.stack([shape[0]*2,shape[1]*2]))
    return inference

def de_conv2d(x,W):
    return tf.nn.conv2d_transpose(x,W,output_shape=[batch_size,60,18,5],strides=[1,3,3,1],padding='SAME')


def input_hidden_sem(x):
    with tf.name_scope("sem_conv_layer1"):
        with tf.variable_scope("sem_conv_layer1"):
            w_conv1= weight_variable(name='weights',shape=[5,5,5,64])
            b_conv1= bias_variable(name='bias',shape=[64])
            layer_conv1=tf.nn.relu(conv2d(x,w_conv1)+b_conv1)

    with tf.name_scope("conv_layer1_pooling"):
        pooling_layer1=tf.nn.max_pool(layer_conv1,ksize=[1,2,2,1],
                                       strides=[1,2,2,1],padding='SAME')

    with tf.name_scope("sem_flat"):
        with tf.variable_scope("sem_flat"):
            w_fc=weight_variable(name='weights',shape=[10*3*64,1024])
            b_fc=bias_variable(name='bias',shape=[1024])
            hidden_layer=tf.nn.relu(tf.matmul(tf.reshape(pooling_layer1,[-1,10*3*64]),w_fc)+b_fc)
    return hidden_layer

def hidden_output_sem(x):
    with tf.name_scope("sem_flat_to_deconv"):
        with tf.variable_scope("sem_flat_to_deconv"):
            w_flat_deconv=weight_variable(name='weights',shape=[1024,10*3*64])
            b_flat_deconv=bias_variable(name='bias',shape=[10*3*64])
            de_flat_layer=tf.nn.relu(tf.matmul(x,w_flat_deconv)+b_flat_deconv)
            de_layer1=tf.reshape(de_flat_layer,[-1,10,3,64])

        unpooling_layer1=max_unpool_2x2(de_layer1,shape=[10,3])

    with tf.name_scope("sem_deconv_output"):
        with tf.variable_scope("sem_deconv_output"):
            w_deconv=weight_variable(name='weights',shape=[5,5,5,64])
            b_deconv=weight_variable(name='bias',shape=[5])
            out=tf.nn.relu(de_conv2d(unpooling_layer1,w_deconv)+b_deconv)
    return out

def sem_loss(y_label,y_pred):

	loss=-(tf.multiply(y_label,tf.log(tf.clip_by_value(y_pred,1e-10,1) ) )
		   +tf.multiply(1-y_label, tf.log( tf.clip_by_value(1-y_pred,1e-10,1) ) )
		   )
	loss=tf.reduce_sum(loss)

	return loss


#  build the model 
with tf.name_scope("sem_input"):
        x = tf.placeholder(tf.float32, shape=[None,60,18,5])

h_fc=input_hidden_sem(x)
with tf.name_scope("sem_dropout"):
    with tf.variable_scope("sem_dropout"):
        keep_prob=tf.placeholder(tf.float32)
        h_fc_drop=tf.nn.dropout(h_fc,keep_prob)
out=hidden_output_sem(h_fc_drop)


with tf.name_scope("loss"):
    #loss=tf.reduce_sum(tf.nn.l2_loss(out-x))
    loss=sem_loss(x,out)


with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

train_size=data.shape[0]
train_indices=range(train_size)
init=tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction =0.4
saver=tf.train.Saver()
sem_path="../sem_model"
if not os.path.isdir(sem_path):
    os.mkdir(sem_path)

with tf.Session(config=config) as sess:
    sess.run(init)
    for ipoch in range(10):
        perm_indices=np.random.permutation(train_indices)
        for step in range(int(train_size/batch_size)):

            offset=(step*batch_size)%(train_size-batch_size)
            batch_indices=perm_indices[offset:(offset+batch_size)]

            _,l=sess.run([train_step,loss],feed_dict={x:data[batch_indices],keep_prob:0.5})
        saver.save(sess,sem_path+'/sem.ckpt')
        print('ipoch:',ipoch ,'loss:',l)
