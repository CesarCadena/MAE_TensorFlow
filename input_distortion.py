import tensorflow as tf
import numpy as np



def input_distortion(imr,img,imb,depth,gnd,obj,bld,veg,sky,border1,border2,resolution,singleframe=False):

    n_frames = len(imr)
    if singleframe==True:
        n_frames = 1

    distorted_outputs = [[],
                         [],
                         [],
                         [],
                         [],
                         [],
                         [],
                         [],
                         []] # create a list with empty lists for every channel


    for i in range(0,n_frames):
        u = np.random.uniform(0,1,1)

        if u < border1:
            if singleframe == False:
                c_1,c_2,c_3,c_4,c_5,c_6,c_7,c_8,c_9 = random_distortion(imr[i],img[i],imb[i],depth[i],gnd[i],obj[i],bld[i],veg[i],sky[i],resolution)
            if singleframe == True:
                c_1,c_2,c_3,c_4,c_5,c_6,c_7,c_8,c_9 = random_distortion(imr,img,imb,depth,gnd,obj,bld,veg,sky,resolution)
            c_distorted = [c_1,c_2,c_3,c_4,c_5,c_6,c_7,c_8,c_9]
            for j in range(0,len(c_distorted)):
                distorted_outputs[j].append(c_distorted[j])

        if u >= border1 and u < border2:
            if singleframe == False:
                c_1,c_2,c_3,c_4,c_5,c_6,c_7,c_8,c_9 = only_rgb(imr[i],img[i],imb[i],depth[i],gnd[i],obj[i],bld[i],veg[i],sky[i],resolution)
            if singleframe == True:
                c_1,c_2,c_3,c_4,c_5,c_6,c_7,c_8,c_9 = only_rgb(imr,img,imb,depth,gnd,obj,bld,veg,sky,resolution)
            c_distorted = [c_1,c_2,c_3,c_4,c_5,c_6,c_7,c_8,c_9]
            for j in range(0,len(c_distorted)):
                distorted_outputs[j].append(c_distorted[j])

        if u >= border2:
            if singleframe == False:
                c_distorted = [imr[i],img[i],imb[i],depth[i],gnd[i],obj[i],bld[i],veg[i],sky[i]]
            if singleframe == True:
                c_distorted = [imr,img,imb,depth,gnd,obj,bld,veg,sky]
            for j in range(0,len(c_distorted)):
                distorted_outputs[j].append(c_distorted[j])


    return distorted_outputs[0],distorted_outputs[1],distorted_outputs[2],distorted_outputs[3],distorted_outputs[4],distorted_outputs[5],distorted_outputs[6],distorted_outputs[7],distorted_outputs[8]



def random_distortion(imr,img,imb,depth,gnd,obj,bld,veg,sky,resolution):

    channels = [imr,img,imb,depth,gnd,obj,bld,veg,sky]

    channels_dist = []
    n_channels = len(channels)

    n_pixels = resolution[0]*resolution[1] # resolution needs to be a tuple with two entries
    noise = 0.1

    for i in range(0,n_channels):
        indices = np.random.choice(n_pixels,int(noise*n_pixels),replace=False)
        im =np.asarray(channels[i])
        np.put(im,indices,0)
        im = im.tolist()
        channels_dist.append(im)

    return channels_dist[0],channels_dist[1],channels_dist[2],channels_dist[3],channels_dist[4],channels_dist[5],channels_dist[6],channels_dist[7],channels_dist[8]



def only_rgb(imr,img,imb,depth,gnd,obj,bld,veg,sky,resolution):

    n_pixels = resolution[0]*resolution[1]
    depth_dist = [0]*n_pixels
    gnd_dist = [0]*n_pixels
    obj_dist = [0]*n_pixels
    bld_dist = [0]*n_pixels
    veg_dist = [0]*n_pixels
    sky_dist = [0]*n_pixels

    return imr,img,imb,depth_dist,gnd_dist,obj_dist,bld_dist,veg_dist,sky_dist
