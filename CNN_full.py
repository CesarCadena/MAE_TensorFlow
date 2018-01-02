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
#%matplotlib inline 
# display inline

#  load data 
data_depth=np.load('../depth_data.npy')
data_depth=np.transpose(data_depth,(0,2,1,3))# swap the two dimensions
print(data_depth.shape)

data_sem=np.load('../sem_data.npy')
data_sem=np.transpose(data_sem,(0,2,1,3))# swap the two dimensions
print(data_sem.shape)

data_rgb=np.load('../rgb_data.npy')
data_rgb=np.transpose(data_rgb,(0,2,1,3))# swap the two dimensions
print(data_rgb.shape)


# Encoder

inputs_depth= tf.placeholder(tf.float32,(None, 18,60,1), name="input_depth")
outputs_depth=tf.placeholder(tf.float32,(None, 18,60,1), name="ouput_depth")

inputs_sem= tf.placeholder(tf.float32,(None, 18,60,5), name="input_sem")
outputs_sem=tf.placeholder(tf.float32,(None, 18,60,5), name="ouput_sem")

inputs_rgb=tf.placeholder(tf.float32,(None, 18,60,3), name="input_rgb")
outputs_rgb=tf.placeholder(tf.float32,(None, 18,60,3), name="ouput_rgb")

############################## Depth #############################
conv1_depth=tf.layers.conv2d(inputs=inputs_depth,filters=16,kernel_size=(3,3),padding='same',
                       activation=tf.nn.relu)
#now (batch,18,60,16)

pool1_depth=tf.layers.max_pooling2d(conv1_depth,pool_size=(2,2),strides=(2,2),padding='same')
#now (batch,9,30,16)

conv2_depth=tf.layers.conv2d(inputs=pool1_depth,filters=8,kernel_size=(3,3),padding='same',
                       activation=tf.nn.relu)
# now (batch,9,30,8)

pool2_depth=tf.layers.max_pooling2d(conv2_depth,pool_size=(2,2),strides=(2,2),padding='same')
#now (batch,5,15,8)

################################## Semantic #################################################

conv1_sem=tf.layers.conv2d(inputs=inputs_sem,filters=16,kernel_size=(3,3),padding='same',
                       activation=tf.nn.relu)
#now (batch,18,60,16)

pool1_sem=tf.layers.max_pooling2d(conv1_sem,pool_size=(2,2),strides=(2,2),padding='same')
#now (batch,9,30,16)

conv2_sem=tf.layers.conv2d(inputs=pool1_sem,filters=8,kernel_size=(3,3),padding='same',
                       activation=tf.nn.relu)
# now (batch,9,30,8)

pool2_sem=tf.layers.max_pooling2d(conv2_sem,pool_size=(2,2),strides=(2,2),padding='same')
#now (batch,5,15,8)
####################################  RGB  ###############################################
conv1_rgb=tf.layers.conv2d(inputs=inputs_rgb,filters=16,kernel_size=(3,3),padding='same',
                       activation=tf.nn.relu)
#now (batch,18,60,16)

pool1_rgb=tf.layers.max_pooling2d(conv1_rgb,pool_size=(2,2),strides=(2,2),padding='same')
#now (batch,9,30,16)

conv2_rgb=tf.layers.conv2d(inputs=pool1_rgb,filters=8,kernel_size=(3,3),padding='same',
                       activation=tf.nn.relu)
# now (batch,9,30,8)

pool2_rgb=tf.layers.max_pooling2d(conv2_rgb,pool_size=(2,2),strides=(2,2),padding='same')
#now (batch,5,15,8)



####Modalities Fusion

flat_hidden_sem=tf.reshape(pool2_sem,[-1,600])
flat_hidden_depth=tf.reshape(pool2_depth,[-1,600])
flat_hidden_rgb=tf.reshape(pool2_rgb,[-1,600])
########## Define Parameters #########
weights_in=tf.Variable(tf.random_normal(shape=[1800,1024],
                                   stddev=0.01),name="weights_in")
bias_in=tf.Variable(tf.zeros([1,1024]),name="bias_in")

# full shared units
flat_full=tf.concat([flat_hidden_rgb,flat_hidden_depth,flat_hidden_sem],axis=1)
full_hidden=tf.nn.relu(tf.matmul(flat_full,weights_in)+bias_in)

weights_out=tf.Variable(tf.random_normal(shape=[1024,1800],
                                   stddev=0.01),name="weights_in")
bias_out=tf.Variable(tf.zeros([1,1800]),name="bias_in")

flat_full_out=tf.nn.relu(tf.matmul(full_hidden,weights_out)+bias_out)


hidden_rgb_out,hidden_depth_out,hidden_sem_out=tf.split(flat_full_out,num_or_size_splits=3,axis=1)


pool2_depth_out=tf.reshape(hidden_depth_out,[-1,5,15,8])
pool2_sem_out=tf.reshape(hidden_sem_out,[-1,5,15,8])
pool2_rgb_out=tf.reshape(hidden_rgb_out,[-1,5,15,8])

##Decoder

### Decoder using high level modules 
upsample1_depth=tf.image.resize_images(pool2_depth_out,size=(9,30),
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# now (batch,9,30,8)
conv4_depth=tf.layers.conv2d(inputs=upsample1_depth,filters=16,kernel_size=(3,3),padding='same',
                       activation=tf.nn.relu)
#now (batch,9,30,8)

upsample2_depth= tf.image.resize_images(conv4_depth, size=(18,60),
                                   method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
#now (batch,18,60,8)
out_depth=tf.layers.conv2d(inputs=upsample2_depth,filters=1,kernel_size=(3,3),padding='same',
                       activation=tf.nn.relu)

################################################

upsample1_sem=tf.image.resize_images(pool2_sem_out,size=(9,30),
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# now (batch,9,30,8)
conv4_sem=tf.layers.conv2d(inputs=upsample1_sem,filters=16,kernel_size=(3,3),padding='same',
                       activation=tf.nn.relu)
#now (batch,9,30,8)

upsample2_sem= tf.image.resize_images(conv4_sem, size=(18,60),
                                   method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
#now (batch,18,60,8)
out_sem=tf.layers.conv2d(inputs=upsample2_sem,filters=5,kernel_size=(3,3),padding='same',
                       activation=tf.nn.relu)
##################################################


upsample1_rgb=tf.image.resize_images(pool2_rgb_out,size=(9,30),
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# now (batch,9,30,8)
conv4_rgb=tf.layers.conv2d(inputs=upsample1_rgb,filters=16,kernel_size=(3,3),padding='same',
                       activation=tf.nn.relu)
#now (batch,9,30,8)

upsample2_rgb= tf.image.resize_images(conv4_rgb, size=(18,60),
                                   method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
#now (batch,18,60,8)
out_rgb=tf.layers.conv2d(inputs=upsample2_rgb,filters=3,kernel_size=(3,3),padding='same',
                       activation=tf.nn.relu)


###Define Loss

# define loss
learning_rate=1e-4
loss=(tf.nn.l2_loss(out_rgb-outputs_rgb)
     +tf.nn.l2_loss(out_sem-outputs_sem)
     +tf.nn.l2_loss(out_depth-outputs_depth)
     )
optimizer=tf.train.AdamOptimizer(learning_rate).minimize(loss)

init=tf.global_variables_initializer()

train_size=data_depth.shape[0]
train_indices=range(train_size)   

CNN_path="../CNN_full"
if not os.path.isdir(CNN_path):
    os.mkdir(CNN_path)          
saver=tf.train.Saver()

# training process
with tf.Session() as sess:
    sess.run(init)
    for ipochs in range(30):
        perm_indices=np.random.permutation(train_indices)
        for step in range(int(train_size/batch_size)):
            offset=(step*batch_size)%(train_size-batch_size)
            batch_indices=perm_indices[offset:(offset+batch_size)]
            
            l,_=sess.run([loss,optimizer],feed_dict={inputs_depth:data_depth[batch_indices],
                                                     outputs_depth:data_depth[batch_indices],
                                                     inputs_rgb:data_rgb[batch_indices],
                                                     outputs_rgb:data_rgb[batch_indices],
                                                     inputs_sem:data_sem[batch_indices],
                                                     outputs_sem:data_sem[batch_indices]
                                                    })        
            print("ipoch: {} ".format(ipochs) ,
                  "step: {}...".format(step),
                  "Training loss: {:.4f}".format(l)) 
        saver.save(sess,CNN_path+'/cnn.ckpt') 
