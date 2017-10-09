
#this code is to connect sequencial images to a video

import tensorflow as tf
import numpy as np
import pandas as pd
import scipy.io
from PIL import Image
import os

flag='validation'

data_index=pd.read_csv('../MAE_KITTI/kitti_split.txt',header=None)


if flag=='validation':
    input_index=data_index[59:61]
elif flag=='training':
    input_index=data_index[1:29]
elif flag=='test':
    input_index=data_index[31:58]
else:
    print('error')


foldernamme='../MAE_KITTI/data_18x60/'
input_index=input_index.values



video_path="../img_to_video/"
if not os.path.isdir(video_path):
    os.mkdir(video_path)


for i in range(1,2):#len(input_index)):

	name=input_index[i][0]

	RGB1name=foldernamme+'data_kitti_im02_'+name+'_18x60.mat'
	RGB2name=foldernamme+'data_kitti_im03_'+name+'_18x60.mat'
	mat1=scipy.io.loadmat(RGB1name)
	mat2=scipy.io.loadmat(RGB2name)

	lenth=mat1['xcr'].shape[1]
	for j in range(lenth):

		img1=np.zeros((18,60,3)).astype(np.uint8)
		img1[:,:,0]=mat1['xcr'][:,j].reshape(60,18).transpose()
		img1[:,:,1]=mat1['xcg'][:,j].reshape(60,18).transpose()
		img1[:,:,2]=mat1['xcb'][:,j].reshape(60,18).transpose()

		img1=Image.fromarray(img1)
		img1.save(video_path+'img%04d.png'%j)



