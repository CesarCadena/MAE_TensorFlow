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

print("loading data.....")
#  prepare data 
data=process_data('training')
Red_data=data['Red']
Green_data=data['Green']
Blue_data=data['Blue']


Red_data=np.reshape(Red_data,[-1,60,18,1])
Blue_data=np.reshape(Blue_data,[-1,60,18,1])
Green_data=np.reshape(Green_data,[-1,60,18,1])
data=np.concatenate([Red_data,Blue_data,Green_data],axis=3)
print(data.shape)



# build the model 
# define functions
def weight_variable(name,shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.get_variable(name=name,dtype=tf.float32,
                           initializer=initial,trainable=True)
def bias_variable(name,shape):
    initial=0.1+tf.zeros(shape)
    return tf.get_variable(name=name,dtype=tf.float32,
                          initializer=initial,trainable=True)
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def de_conv2d(x,W):
    return tf.nn.conv2d_transpose(x,W,strides=[1,1,1,1],padding='SAME')


#
with tf.name_scope("input"):
    x = tf.placeholder(tf.float32, shape=[None,60,18,3])


with tf.name_scope("conv_layer1"):
	with tf.variable_scope("conv_layer1"):
		W_conv1= weight_variable(name='weights',shape=[5, 5, 1, 32])
		b_conv1= bias_variable(name='bias',shape=[32])

		layer_conv1=tf.nn.relu(conv2d(x,W_conv1)+b_conv1)


with tf.name_scope("conv_layer2"):
	with tf.variable_scope("conv_layer2"):
		 W_conv2=weight_variable(name='weights',shape=[5,5,32,64])
         b_conv2=bias_variable('bias',shape=[64])
         layer_conv2=tf.nn.relu(conv2d(layer_pool1,W_conv2)+b_conv2)


with tf.name_scope("de_conv_layer2"):
	with tf.variable_scope("de_conv_layer2"):
		 W_deconv2=weight_variable(name='weights',shape=[5,5,64,32])
         b_deconv2=bias_variable('bias',shape=[32])
         layer_deconv2=tf.nn.relu(de_conv2d(layer_conv2,W_deconv2)+b_deconv2)        

with tf.name_scope("de_conv_layer1"):
	with tf.variable_scope("de_conv_layer1"):
		 W_deconv1=weight_variable(name='weights',shape=[5,5,32,3])
         b_deconv1=bias_variable('bias',shape=[3])
         layer_deconv1=tf.nn.relu(de_conv2d(layer_deconv2,W_deconv1)+b_deconv1) 


with tf.name_scope("loss"):
	loss=tf.nn.l2_loss(layer_deconv1-x)
with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)



train_size=data.shape[0]
train_indices=range(train_size)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction =0.5


with tf.Session(config=config) as sess:

	for ipoch in range(2):

		perm_indices=np.random.permutation(train_indices)

		for step in range(int(train_size/batch_size)):
			offset=(step*batch_size)%(train_size-batch_size)
            batch_indices=perm_indices[offset:(offset+batch_size)]

            sess.run(train_step,feed_dict={x:data})
