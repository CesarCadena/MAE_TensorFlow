import numpy as np
from copy import copy



def create(n, constructor=list):
    '''
    this function creates a list containing n containers of type constructor
    use: a = list(create(10,constructor=list)), this yields a list of 10 empty lists
    :param n: size
    :param constructor: type of container
    :return:
    '''
    for _ in range(n):
        yield constructor()

def invert_depth(inv_depth):

    inv_depth = np.reshape(inv_depth,(max(inv_depth.shape),1))
    depth = np.zeros(inv_depth.shape)

    for i in range(0,inv_depth.shape[0]):
        if inv_depth[i] > 1/80.:
            depth[i] = 1./inv_depth[i]
    return depth

def horizontal_mirroring(x,ind_who,resolution = (18,60),option=None):

    if option == None:
        h = resolution[0]
        w = resolution[1]
        indmirror = np.linspace(0,h*w-1,h*w).astype(int)
        indmirror = np.reshape(indmirror,(w,h))
        indmirror = indmirror.T
        indmirror = np.fliplr(indmirror)
        indmirror = indmirror.T
        indmirror = np.reshape(indmirror,(h*w))

        x1 = copy(x)
        x = np.array(x)
        for i in ind_who:
            x1[i]=x[i][indmirror]

        return x1

    if option == 'RNN':

        n_steps = len(x[0])

        h = resolution[0]
        w = resolution[1]
        indmirror = np.linspace(0,h*w-1,h*w).astype(int)
        indmirror = np.reshape(indmirror,(w,h))
        indmirror = indmirror.T
        indmirror = np.fliplr(indmirror)
        indmirror = indmirror.T
        indmirror = np.reshape(indmirror,(h*w))

        x1 = copy(x)
        x = np.array(x)
        for i in ind_who:
            for rnn_step in range(0,n_steps):
                x1[i][rnn_step]=x[i][rnn_step][indmirror]

        return x1


def zeroing_channel(x,ind_who,resolution = (18,60)):

    h = resolution[0]
    w = resolution[1]

    x1 = copy(x)
    zeros = np.zeros((h*w))

    for i in ind_who:
        x1[i] = zeros

    return x1

def color2image(red_channel,green_channel,blue_channel,resolution=(18,60)):

    normalization = 1.
    if max(red_channel) > 1.:
        normalization = 255.

    imr = np.reshape(red_channel/normalization,(resolution[1],resolution[0])).T
    img = np.reshape(green_channel/normalization,(resolution[1],resolution[0])).T
    imb = np.reshape(blue_channel/normalization,(resolution[1],resolution[0])).T
    im = np.dstack((imr,img,imb))

    return im

def depth2image(depth_channel,mask_channel,resolution=(18,60)):

    imd = np.reshape(depth_channel,(resolution[1],resolution[0])).T * np.reshape(mask_channel,(resolution[1],resolution[0])).T

    return imd

def sem2image(semantic_channel,resolution=(18,60)):

    ims = np.reshape(semantic_channel,(resolution[1],resolution[0])).T

    return ims

def get_frames(sequences,data,size_input=1080):

        imr_data = []
        img_data = []
        imb_data = []
        dpt_data = []
        dpt_msk_data = []
        gnd_data = []
        obj_data = []
        bld_data = []
        veg_data = []
        sky_data= []

        zero_padding = np.zeros((size_input,))


        for sequence in sequences:

            imr_seq = []
            img_seq = []
            imb_seq = []
            dpt_seq = []
            dpt_msk_seq = []
            gnd_seq = []
            obj_seq = []
            bld_seq = []
            veg_seq = []
            sky_seq = []

            for frame in sequence:

                if frame[0] == -1:

                    imr_seq.append(zero_padding)
                    img_seq.append(zero_padding)
                    imb_seq.append(zero_padding)
                    dpt_seq.append(zero_padding)
                    dpt_msk_seq.append(zero_padding)
                    gnd_seq.append(zero_padding)
                    obj_seq.append(zero_padding)
                    bld_seq.append(zero_padding)
                    veg_seq.append(zero_padding)
                    sky_seq.append(zero_padding)

                else:

                    seq = frame[0]
                    frm = frame[1]

                    imr_seq.append(data[seq][frm]['xcr1']/255.)
                    img_seq.append(data[seq][frm]['xcg1']/255.)
                    imb_seq.append(data[seq][frm]['xcb1']/255.)
                    dpt_seq.append(data[seq][frm]['xid1'])
                    dpt_msk_seq.append(data[seq][frm]['xmask1'])
                    gnd_seq.append((data[seq][frm]['sem1']==1).astype(int))
                    obj_seq.append((data[seq][frm]['sem1']==2).astype(int))
                    bld_seq.append((data[seq][frm]['sem1']==3).astype(int))
                    veg_seq.append((data[seq][frm]['sem1']==4).astype(int))
                    sky_seq.append((data[seq][frm]['sem1']==5).astype(int))


            imr_data.append(copy(imr_seq))
            img_data.append(copy(img_seq))
            imb_data.append(copy(imb_seq))
            dpt_data.append(copy(dpt_seq))
            dpt_msk_data.append(copy(dpt_msk_seq))
            gnd_data.append(copy(gnd_seq))
            obj_data.append(copy(obj_seq))
            bld_data.append(copy(bld_seq))
            veg_data.append(copy(veg_seq))
            sky_data.append(copy(sky_seq))

        return [imr_data,img_data,imb_data,dpt_data,dpt_msk_data,gnd_data,obj_data,bld_data,veg_data,sky_data]






