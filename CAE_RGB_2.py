import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from process_data import  process_data
batch_size=20
num_epochs=100
filter_size=16
RESTORE=0
SEED = None
#%matplotlib inline 
# display inline
data=np.load('../Data/rgb_data.npy')
print(data.shape)

with tf.variable_scope("RGB"):
    
    inputs= tf.placeholder(tf.float32, (None, 18,60,3), name="input")
    outputs=tf.placeholder(tf.float32, (None, 18,60,3), name="ouput")
    ### Encoder use high level module 
    conv1=tf.layers.conv2d(inputs=inputs,filters=2*filter_size,kernel_size=(3,3),padding='same',
                       activation=tf.nn.relu,name='conv1')
#now (batch,18,60,16)

    pool1=tf.layers.max_pooling2d(conv1,pool_size=(2,2),strides=(2,2),padding='same')
#now (batch,9,30,16)

    conv2=tf.layers.conv2d(inputs=pool1,filters=filter_size,kernel_size=(3,3),padding='same',
                       activation=tf.nn.relu,name='conv2')
# now (batch,9,30,8)

    pool2=tf.layers.max_pooling2d(conv2,pool_size=(2,2),strides=(2,2),padding='same')
#now (batch,5,15,8)
#########################################################################################
    ### Decoder using high level modules 
    upsample1=tf.image.resize_images(pool2,size=(9,30),
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# now (batch,9,30,8)
    conv4=tf.layers.conv2d(inputs=upsample1,filters=2*filter_size,kernel_size=(3,3),padding='same',
                       activation=tf.nn.relu,name='conv4')
#now (batch,9,30,8)

    upsample2 = tf.image.resize_images(conv4, size=(18,60),
                                   method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
#now (batch,18,60,8)
    out=tf.layers.conv2d(inputs=upsample2,filters=3,kernel_size=(3,3),padding='same',
                       activation=tf.nn.sigmoid,name='out')
#now (batch,18,60,3)

learning_rate=1e-3
loss=tf.nn.l2_loss(out-outputs)
optimizer=tf.train.AdamOptimizer(learning_rate).minimize(loss)

init=tf.global_variables_initializer()
train_size=data.shape[0]
train_indices=range(train_size)
saver=tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction =0.4


with tf.Session(config=config) as sess:
    sess.run(init)
    for ipochs in range(num_epochs):
        perm_indices=np.random.permutation(train_indices)

        for step in range(int(train_size/batch_size)):
            offset=(step*batch_size)%(train_size-batch_size)
            batch_indices=perm_indices[offset:(offset+batch_size)]
            l,_=sess.run([loss,optimizer],feed_dict={inputs:data[batch_indices],
                                                     outputs:data[batch_indices]})
            
        print("Epoch: {}...".format(ipochs),
                       "Training loss: {:.4f}".format(l))
        saver.save(sess,'./CNN_models/rgb_100_16/rgb.ckpt')
#test on reconstruction  
    #reconstruction=sess.run(out,feed_dict={inputs:data[0:1]})
    
#plt.imshow(reconstruction[0])