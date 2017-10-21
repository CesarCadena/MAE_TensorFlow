import tensorflow as tf
import numpy as np
import basic_routines as BR
from copy import copy


def pretraining_input_distortion(im,resolution=(18,60),singleframe=False):
    n_frames = len(im)

    if singleframe==True:
        n_frames = 1

    distorted_outputs = []

    for i in range(0,n_frames):
        if singleframe == False:
                im_dist = random_distortion2(im[i],resolution)
        if singleframe == True:
                im_dist = random_distortion2(im,resolution)

        distorted_outputs.append(copy(im_dist))

    return distorted_outputs




def input_distortion(imr,img,imb,depth,gnd,obj,bld,veg,sky,resolution,rnn=False,singleframe=False):


    if rnn == True:

        n_channels = 9 # number of channels in the MAE
        n_frames = len(imr) # batch size
        n_steps = len(imr[0]) # number of recurrent steps

        if singleframe==True:
            n_frames = 1 # single frame (often used in validation)
            n_steps = len(imr) # number of recurrent steps

        inputs_dist = list(BR.create(n_channels,constructor=list)) # creates a list of empty lists for each channel

        for i in range(0,n_frames):

            series = list(BR.create(n_channels,constructor=list))

            for k in range(0,n_steps):

                if singleframe == False:

                    frames_dist = random_distortion(imr[i][k],
                                                    img[i][k],
                                                    imb[i][k],
                                                    depth[i][k],
                                                    gnd[i][k],
                                                    obj[i][k],
                                                    bld[i][k],
                                                    veg[i][k],
                                                    sky[i][k],
                                                    resolution) # random distortion of the input
                if singleframe == True:

                    frames_dist = random_distortion(imr[k],
                                                    img[k],
                                                    imb[k],
                                                    depth[k],
                                                    gnd[k],
                                                    obj[k],
                                                    bld[k],
                                                    veg[k],
                                                    sky[k],
                                                    resolution) # random distortion of the input

                for j in range(0,n_channels):
                    series[j].append(frames_dist[j])

            for j in range(0,n_channels):
                inputs_dist[j].append(series[j])

        # returning the distorted inputs (one per channel)
        return inputs_dist[0],inputs_dist[1],inputs_dist[2],inputs_dist[3],inputs_dist[4],inputs_dist[5],inputs_dist[6],inputs_dist[7],inputs_dist[8]


    else:

        n_channels = 9
        n_frames = len(imr)

        if singleframe==True:
            n_frames = 1

        inputs_dist = list(BR.create(n_channels,constructor=list))


        for i in range(0,n_frames):

            if singleframe == False:
                frames_dist = random_distortion(imr[i],
                                                img[i],
                                                imb[i],
                                                depth[i],
                                                gnd[i],
                                                obj[i],
                                                bld[i],
                                                veg[i],
                                                sky[i],
                                                resolution)

            if singleframe == True:
               frames_dist = random_distortion(imr,img,imb,depth,gnd,obj,bld,veg,sky,resolution)


            for j in range(0,n_channels):
                inputs_dist[j].append(frames_dist[j])

        return inputs_dist[0],inputs_dist[1],inputs_dist[2],inputs_dist[3],inputs_dist[4],inputs_dist[5],inputs_dist[6],inputs_dist[7],inputs_dist[8]


def random_distortion2(im,resolution):
    n_pixels = resolution[0]*resolution[1] # resolution needs to be a tuple with two entries
    noise = 0.1
    indices = np.random.choice(n_pixels,int(noise*n_pixels),replace=False)
    im_dist =np.asarray(im)
    np.put(im_dist,indices,0)
    im_dist = im_dist.tolist()
    im_dist1 = copy(im_dist)
    return im_dist1

def random_distortion(imr,img,imb,depth,gnd,obj,bld,veg,sky,resolution):

    channels = [imr,img,imb,depth,gnd,obj,bld,veg,sky]

    channels_dist = []
    n_channels = len(channels)

    n_pixels = resolution[0]*resolution[1] # resolution needs to be a tuple with two entries
    noise = 0.1

    for i in range(0,n_channels):
        indices = np.random.choice(n_pixels,int(noise*n_pixels),replace=False)
        im = copy(np.asarray(channels[i]))
        im = im.tolist()
        im1 = copy(im)
        channels_dist.append(copy(im1))

    return channels_dist


def only_rgb(imr,img,imb,depth,gnd,obj,bld,veg,sky,resolution):

    n_pixels = resolution[0]*resolution[1]
    depth_dist = [0]*n_pixels
    gnd_dist = [0]*n_pixels
    obj_dist = [0]*n_pixels
    bld_dist = [0]*n_pixels
    veg_dist = [0]*n_pixels
    sky_dist = [0]*n_pixels

    return imr,img,imb,depth_dist,gnd_dist,obj_dist,bld_dist,veg_dist,sky_dist
