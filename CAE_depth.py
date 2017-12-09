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

data=process_data('training')
Depth_data=data['Depth']
Depthmask_data=data['Depthmask']
Depth_data=np.multiply(Depth_data,Depthmask_data)
Depth_data=np.reshape(Depth_data,[-1,60,18,1])
#np.save('../depth(num,60,18,1)',Depth_data)

#data=np.load('../depth(num,60,18,1).npy')
#print(data.shape)

## build handy functions
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
    return tf.nn.conv2d_transpose(x,W,output_shape=[batch_size,60,18,1],strides=[1,3,3,1],padding='SAME')


###build the model 
def input_hidden_depth(x):
    with tf.name_scope("depth_conv_layer1"):
        with tf.variable_scope("depth_conv_layer1"):
            w_conv1= weight_variable(name='weights',shape=[5,5,1,64])
            b_conv1= bias_variable(name='bias',shape=[64])
            layer_conv1=tf.nn.relu(conv2d(x,w_conv1)+b_conv1)

    with tf.name_scope("depth_flat"):
        with tf.variable_scope("depth_flat"):
            w_fc=weight_variable(name='weights',shape=[20*6*64,1024])
            b_fc=bias_variable(name='bias',shape=[1024])
            h_fc=tf.nn.relu(tf.matmul(tf.reshape(layer_conv1,[-1,20*6*64]),w_fc)+b_fc)

    return h_fc


def hidden_output_depth(x):

    with tf.name_scope("depth_flat_to_deconv"):
        with tf.variable_scope("depth_flat_to_deconv"):
            w_flat_deconv=weight_variable(name='weights',shape=[1024,20*6*64])
            b_flat_deconv=bias_variable(name='bias',shape=[20*6*64])
            de_flat_layer=tf.nn.relu(tf.matmul(x,w_flat_deconv)+b_flat_deconv)
            de_layer1=tf.reshape(de_flat_layer,[-1,20,6,64])

    with tf.name_scope("depth_deconv_output"):
        with tf.variable_scope("depth_deconv_output"):
            w_deconv=weight_variable(name='weights',shape=[5,5,1,64])
            b_deconv=weight_variable(name='bias',shape=[1])
            out=tf.nn.relu(de_conv2d(de_layer1,w_deconv)+b_deconv)
    return out




with tf.name_scope("depth_input"):
        x = tf.placeholder(tf.float32, shape=[None,60,18,1])

h_fc=input_hidden_depth(x)

with tf.name_scope("depth_dropout"):
    with tf.variable_scope("depth_dropout"):
        keep_prob=tf.placeholder(tf.float32)
        h_fc_drop=tf.nn.dropout(h_fc,keep_prob)
out=hidden_output_depth(h_fc_drop)


with tf.name_scope("depth_loss"):
    loss=tf.reduce_sum(tf.nn.l2_loss(out-x))

with tf.name_scope("depth_train"):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)


train_size=Depth_data.shape[0]
train_indices=range(train_size)
init=tf.global_variables_initializer()


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction =0.4
saver=tf.train.Saver()
depth_path="../depth_model"
if not os.path.isdir(depth_path):
    os.mkdir(depth_path)


tf.summary.scalar('loss',loss)
summary_op=tf.summary.merge_all()



with tf.Session(config=config) as sess:

    sess.run(init)
    writer=tf.summary.FileWriter('../graphs',sess.graph)

    for ipoch in range(1):
        perm_indices=np.random.permutation(train_indices)

        for step in range(1):#int(train_size/batch_size)):

            offset=(step*batch_size)%(train_size-batch_size)
            batch_indices=perm_indices[offset:(offset+batch_size)]
            
            _,summary=sess.run([train_step,summary_op],feed_dict={x:Depth_data[batch_indices],keep_prob:0.5})
            writer.add_summary(summary,step)

        saver.save(sess,depth_path+'/depth.ckpt')
        print('ipoch:',ipoch)