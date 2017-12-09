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
#prepare the  data 



data=process_data('training')
Red_data=data['Red']
Green_data=data['Green']
Blue_data=data['Blue']
Red_data=np.reshape(Red_data,[-1,60,18,1])
Blue_data=np.reshape(Blue_data,[-1,60,18,1])
Green_data=np.reshape(Green_data,[-1,60,18,1])
data=np.concatenate([Red_data,Blue_data,Green_data],axis=3)
print(data.shape)
#np.save('../rgb(num,60,18,3)',data)


#data=np.load('../rgb(num,60,18,3).npy')
print('finish')
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
    return tf.nn.conv2d(x,W,strides=[1,3,3,1],padding='SAME')


def de_conv2d(x,W):
    return tf.nn.conv2d_transpose(x,W,output_shape=[batch_size,60,18,3],strides=[1,3,3,1],padding='SAME')



def input_hidden_rgb(x):
    with tf.name_scope("rgb_conv_layer1"):
        with tf.variable_scope("rgb_conv_layer1"):
            w_conv1= weight_variable(name='weights',shape=[5,5,3,64])
            b_conv1= bias_variable(name='bias',shape=[64])
            layer_conv1=tf.nn.relu(conv2d(x,w_conv1)+b_conv1)

    with tf.name_scope("rgb_flat"):
        with tf.variable_scope("rgb_flat"):
            w_fc=weight_variable(name='weights',shape=[20*6*64,1024])
            b_fc=bias_variable(name='bias',shape=[1024])
            h_fc=tf.nn.relu(tf.matmul(tf.reshape(layer_conv1,[-1,20*6*64]),w_fc)+b_fc)
    return h_fc


def hidden_output_rgb(x):
    with tf.name_scope("rgb_flat_to_deconv"):
        with tf.variable_scope("rgb_flat_to_deconv"):
            w_flat_deconv=weight_variable(name='weights',shape=[1024,20*6*64])
            b_flat_deconv=bias_variable(name='bias',shape=[20*6*64])
            de_flat_layer=tf.nn.relu(tf.matmul(x,w_flat_deconv)+b_flat_deconv)
            de_layer1=tf.reshape(de_flat_layer,[-1,20,6,64])

    with tf.name_scope("rgb_deconv_output"):
        with tf.variable_scope("rgb_deconv_output"):
            w_deconv=weight_variable(name='weights',shape=[5,5,3,64])
            b_deconv=weight_variable(name='bias',shape=[3])
            out=tf.nn.relu(de_conv2d(de_layer1,w_deconv)+b_deconv)
    return out




with tf.name_scope("input"):
    x = tf.placeholder(tf.float32, shape=[None,60,18,3])
h_fc=input_hidden_rgb(x)
with tf.name_scope("rgb_dropout"):
    with tf.variable_scope("rgb_dropout"):
        keep_prob=tf.placeholder(tf.float32)
        h_fc_drop=tf.nn.dropout(h_fc,keep_prob)
out=hidden_output_rgb(h_fc_drop)


with tf.name_scope("loss"):
    loss=tf.reduce_sum(tf.nn.l2_loss(out-x))

with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)


train_size=data.shape[0]
train_indices=range(train_size)
init=tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction =0.4

saver=tf.train.Saver()

rgb_path="../rgb_model"
if not os.path.isdir(rgb_path):
    os.mkdir(rgb_path)


with tf.Session(config=config) as sess:
    sess.run(init)

    for ipoch in range(10):
        perm_indices=np.random.permutation(train_indices)
        for step in range(int(train_size/batch_size)):

            offset=(step*batch_size)%(train_size-batch_size)
            batch_indices=perm_indices[offset:(offset+batch_size)]

            sess.run(train_step,feed_dict={x:data[batch_indices],keep_prob:0.5})
        saver.save(sess,rgb_path+'/rgb.ckpt')
            print (step)

        print('ipoch:',ipoch)

