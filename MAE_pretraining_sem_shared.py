import tensorflow as tf
import numpy as np
from process_data import process_data
import pandas as pd
import scipy.io
import os
import sys
tf.reset_default_graph()
batch_size=60
hidden_size=1024
num_epoch=100



print("###########loading data ..........########")
FLAG='training'
if (True):
		data=process_data(FLAG)
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
print("###########building model  ..........########")


# shared semantic model 
Ground_input=tf.placeholder(tf.float32,shape=[None,1080])
Objects_input=tf.placeholder(tf.float32,shape=[None,1080])
Building_input=tf.placeholder(tf.float32,shape=[None,1080])
Vegetation_input=tf.placeholder(tf.float32,shape=[None,1080])
Sky_input=tf.placeholder(tf.float32,shape=[None,1080])

"""define encoder layers"""
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

"""define hidden layers"""

Ground_hidden=tf.nn.relu(tf.matmul(Ground_input,Ground_weights)+Ground_bias)
Objects_hidden=tf.nn.relu(tf.matmul(Objects_input,Objects_weights)+Objects_bias)
Building_hidden=tf.nn.relu(tf.matmul(Building_input,Building_weights)+Building_bias)
Vegetation_hidden=tf.nn.relu(tf.matmul(Vegetation_input,Vegetation_weights)+Vegetation_bias)
Sky_hidden=tf.nn.relu(tf.matmul(Sky_input,Sky_weights)+Sky_bias)

"""define semantic layers"""


Semantic_weights=tf.Variable(tf.random_normal(shape=[5*hidden_size,hidden_size],
														 stddev=0.01),name="Semantic_weights")
Semantic_bias=tf.Variable(tf.zeros([1,hidden_size]),name="Semantic_bias")
Semantic_shared=tf.matmul(tf.concat([Ground_hidden,Objects_hidden,Building_hidden,Vegetation_hidden,Sky_hidden],1)
																		 ,Semantic_weights)+Semantic_bias


Semantic_deweights=tf.Variable(tf.random_normal(shape=[hidden_size,5*hidden_size],
																					 stddev=0.01),name="Semantic_deweights")
Semantic_debias=tf.Variable(tf.zeros([1,5*hidden_size]),name="Semantic_debias")  
Semantic_deshared=tf.matmul(Semantic_shared,Semantic_deweights)+Semantic_debias

decoder_ground,decoder_objects,decoder_building,decoder_vegetation,decoder_sky=tf.split(Semantic_deshared,num_or_size_splits=5, axis=1)

"""define decoder layers"""

Ground_outweights=tf.Variable(tf.random_normal(shape=[hidden_size,1080],
																	 stddev=0.01),name="Ground_outweights")
Ground_outbias=tf.Variable(tf.zeros([1,1080]),name="Ground_outbias")
Ground_out=tf.nn.sigmoid(tf.matmul(decoder_ground,Ground_outweights)+Ground_outbias)



Objects_outweights=tf.Variable(tf.random_normal(shape=[hidden_size,1080],
																	 stddev=0.01),name="Objects_outweights")
Objects_outbias=tf.Variable(tf.zeros([1,1080]),name="Objects_outbias")
Objects_out=tf.nn.sigmoid(tf.matmul(decoder_objects,Objects_outweights)+Objects_outbias)



Building_outweights=tf.Variable(tf.random_normal(shape=[hidden_size,1080],
																	 stddev=0.01),name="Building_outweights")
Building_outbias=tf.Variable(tf.zeros([1,1080]),name="Building_outbias")
Building_out=tf.nn.sigmoid(tf.matmul(decoder_building,Building_outweights)+Building_outbias)



Vegetation_outweights=tf.Variable(tf.random_normal(shape=[hidden_size,1080],
																	 stddev=0.01),name="Vegetation_outweights")
Vegetation_outbias=tf.Variable(tf.zeros([1,1080]),name="Vegetation_outbias")
Vegetation_out=tf.nn.sigmoid(tf.matmul(decoder_vegetation,Vegetation_outweights)+Vegetation_outbias)


Sky_outweights=tf.Variable(tf.random_normal(shape=[hidden_size,1080],
																	 stddev=0.01),name="Sky_outweights")
Sky_outbias=tf.Variable(tf.zeros([1,1080]),name="Sky_outbias")
Sky_out=tf.nn.sigmoid(tf.matmul(decoder_sky,Sky_outweights)+Sky_outbias)

"""define loss"""

loss_sem=-tf.reduce_sum(np.multiply(Ground_input,tf.log(tf.clip_by_value(Ground_out,1e-10,1)))
													+np.multiply(1-Ground_input,tf.log(tf.clip_by_value(1-Ground_out,1e-10,1)))
													)-tf.reduce_sum(np.multiply(Objects_input,tf.log(tf.clip_by_value(Objects_out,1e-10,1)))
													+np.multiply(1-Objects_input,tf.log(tf.clip_by_value(1-Objects_out,1e-10,1)))
													)-tf.reduce_sum(np.multiply(Building_input,tf.log(tf.clip_by_value(Building_out,1e-10,1)))
													+np.multiply(1-Building_input,tf.log(tf.clip_by_value(1-Building_out,1e-10,1)))
													)-tf.reduce_sum(np.multiply(Sky_input,tf.log(tf.clip_by_value(Sky_out,1e-10,1)))
													+np.multiply(1-Sky_input,tf.log(tf.clip_by_value(1-Sky_out,1e-10,1)))
													)-tf.reduce_sum(np.multiply(Vegetation_input,tf.log(tf.clip_by_value(Vegetation_out,1e-10,1)))
													+np.multiply(1-Vegetation_input,tf.log(tf.clip_by_value(1-Vegetation_out,1e-10,1)))
													)
				
regularization_sem=(tf.nn.l2_loss(Ground_weights)
									 +tf.nn.l2_loss(Objects_weights)
									 +tf.nn.l2_loss(Building_weights)
									 +tf.nn.l2_loss(Vegetation_weights)
									 +tf.nn.l2_loss(Sky_weights)
									 +tf.nn.l2_loss(Semantic_weights)
									 +tf.nn.l2_loss(Semantic_deweights)
									 +tf.nn.l2_loss(Ground_outweights)
									 +tf.nn.l2_loss(Objects_outweights)
									 +tf.nn.l2_loss(Building_outweights)
									 +tf.nn.l2_loss(Vegetation_outweights)
									 +tf.nn.l2_loss(Sky_outweights))

loss_sem_t=loss_sem+5e-4*regularization_sem


first_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
																 "Sem")

optimizer_sem=tf.train.AdamOptimizer(learning_rate=1e-5,
																			beta1=0.9,beta2=0.999,
																			epsilon=1e-8,use_locking=False,
																			name='Adam').minimize(loss_sem,var_list=first_train_vars)


optimizer_sem_full=tf.train.AdamOptimizer(learning_rate=1e-6,
																			beta1=0.9,beta2=0.999,
																			epsilon=1e-8,use_locking=False,
																			name='Adam').minimize(loss_sem_t)

model_path="../model_sem"
if not os.path.isdir(model_path):
		os.mkdir(model_path)

Restore_list={"Ground_weights": Ground_weights,
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
							"Vegetation_outbias":Vegetation_outbias}


init=tf.global_variables_initializer()
saver=tf.train.Saver(Restore_list)


train_size=Ground_data.shape[0]
train_indices=range(train_size)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5

saver_all=tf.train.Saver()

with tf.Session(config=config) as sess:
		#print('hello!')

		sess.run(init)
 
		saver.restore(sess,'../model_sep/pretraining_sep.ckpt')

		#restore will rewrite the para
		#print(sess.run(Ground_weights)[0][0])
		#print(sess.run(Semantic_weights)[0][0])

		for i in range(30):
				
				perm_indices=np.random.permutation(train_indices)
				
				for step in range(int(train_size/batch_size)):
						
						offset=(step*batch_size)%(train_size-batch_size)
						batch_indices=perm_indices[offset:(offset+batch_size)]
						
						
						feed_dict={Ground_input:Ground_data[batch_indices,:],
											 Objects_input:Objects_data[batch_indices,:],
											 Vegetation_input:Vegetation_data[batch_indices,:],
											 Building_input:Building_data[batch_indices,:],
											 Sky_input:Sky_data[batch_indices,:]
											}
						
						_sem,l_sem,l_rg=sess.run([optimizer_sem,loss_sem,regularization_sem],feed_dict=feed_dict)
		
				print("loss of %d ipoch semantic is :"%i,l_sem,l_rg) 


		for i in range(30,num_epoch):
				
				perm_indices=np.random.permutation(train_indices)
				
				for step in range(int(train_size/batch_size)):
						
						offset=(step*batch_size)%(train_size-batch_size)
						batch_indices=perm_indices[offset:(offset+batch_size)]
						
						
						feed_dict={Ground_input:Ground_data[batch_indices,:],
											 Objects_input:Objects_data[batch_indices,:],
											 Vegetation_input:Vegetation_data[batch_indices,:],
											 Building_input:Building_data[batch_indices,:],
											 Sky_input:Sky_data[batch_indices,:]
											}
						
						_sem,l_sem=sess.run([optimizer_sem_full,loss_sem],feed_dict=feed_dict)

				print("loss of %d ipoch semantic is :"%i,l_sem)      

		saver_all.save(sess,model_path+"/pretraining_sem_full.ckpt")
						
						
		print("training shared sem finished !")
