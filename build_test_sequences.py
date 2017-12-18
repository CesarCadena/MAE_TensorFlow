import numpy as np

from copy import copy


def build_test_sequences(data_test, n_rnn_steps=None,resolution = (18,60)):

    if n_rnn_steps == None:
        raise ValueError('no rnn step size given')

    size_input = resolution[0]*resolution[1]

    imr = []
    img = []
    imb = []
    dpt = []
    dpt_spr = []
    gnd = []
    obj = []
    bld = []
    veg = []
    sky = []


    for sequence in data_test:

        imr_seq = []
        img_seq = []
        imb_seq = []
        dpt_seq = []
        dpt_spr_seq = []
        gnd_seq = []
        obj_seq = []
        bld_seq = []
        veg_seq = []
        sky_seq = []

        s_length = len(sequence)

        for i in range(0,s_length):

            imr_set = []
            img_set = []
            imb_set = []
            dpt_set = []
            dpt_spr_set = []
            gnd_set = []
            obj_set = []
            bld_set = []
            veg_set = []
            sky_set = []

            for step in range(0, n_rnn_steps):

                offset = n_rnn_steps-step
                index = i - offset

                if index < 0:

                    zero_padding = np.zeros((size_input,))
                    imr_set.append(zero_padding)
                    img_set.append(zero_padding)
                    imb_set.append(zero_padding)
                    dpt_set.append(zero_padding)
                    dpt_spr_set.append(zero_padding)
                    gnd_set.append(zero_padding)
                    obj_set.append(zero_padding)
                    bld_set.append(zero_padding)
                    veg_set.append(zero_padding)
                    sky_set.append(zero_padding)
                    continue

                else:
                    imr_set.append(sequence[index]['xcr']/255.)
                    img_set.append(sequence[index]['xcg']/255.)
                    imb_set.append(sequence[index]['xcb']/255.)
                    dpt_set.append(sequence[index]['xid'])
                    dpt_spr_set.append(sequence[index]['xidsparse'])
                    gnd_set.append((sequence[index]['sem']==1).astype(int))
                    obj_set.append((sequence[index]['sem']==2).astype(int))
                    bld_set.append((sequence[index]['sem']==3).astype(int))
                    veg_set.append((sequence[index]['sem']==4).astype(int))
                    sky_set.append((sequence[index]['sem']==5).astype(int))

            imr_seq.append(copy(imr_set))
            img_seq.append(copy(img_set))
            imb_seq.append(copy(imb_set))
            dpt_seq.append(copy(dpt_set))
            dpt_spr_seq.append(copy(dpt_spr_set))
            gnd_seq.append(copy(gnd_set))
            obj_seq.append(copy(obj_set))
            bld_seq.append(copy(bld_set))
            veg_seq.append(copy(veg_set))
            sky_seq.append(copy(sky_set))

        imr.append(copy(imr_seq))
        img.append(copy(img_seq))
        imb.append(copy(imb_seq))
        dpt.append(copy(dpt_seq))
        dpt_spr.append(copy(dpt_spr_seq))
        gnd.append(copy(gnd_seq))
        obj.append(copy(obj_seq))
        bld.append(copy(bld_seq))
        veg.append(copy(veg_seq))
        sky.append(copy(sky_seq))


    return [imr,img,imb,dpt,dpt_spr,gnd,obj,bld,veg,sky]


def distort_test_sequences(test_sequence, n_rnn_steps=None, option=None, frequency=None, resolution = (18,60)):

    if n_rnn_steps == None:
        raise ValueError('no number of rnn steps given')

    if option == None:
        raise ValueError('no data option given')

    if frequency == None:
        raise ValueError('no frequency given')

    size_input = resolution[0]*resolution[1]

    imr = copy(test_sequence[0])
    img = copy(test_sequence[1])
    imb = copy(test_sequence[2])
    dpt = copy(test_sequence[3])
    gnd = copy(test_sequence[5])
    obj = copy(test_sequence[6])
    bld = copy(test_sequence[7])
    veg = copy(test_sequence[8])
    sky = copy(test_sequence[9])

    n_frames = len(imr)

    zeroing = np.zeros(size_input,)

    if option == 'rgbd':

        for frame in range(0,n_frames):
            for step in range(0,n_rnn_steps):

                offset = n_rnn_steps-step
                index = frame - offset

                if index <= 0:
                    continue

                if index%frequency == 0:

                    gnd[frame][step] = zeroing
                    obj[frame][step] = zeroing
                    bld[frame][step] = zeroing
                    veg[frame][step] = zeroing
                    sky[frame][step] = zeroing

                else:

                    dpt[frame][step] = zeroing
                    gnd[frame][step] = zeroing
                    obj[frame][step] = zeroing
                    bld[frame][step] = zeroing
                    veg[frame][step] = zeroing
                    sky[frame][step] = zeroing


    if option == 'rgbs':

        for frame in range(0,n_frames):
            for step in range(0,n_rnn_steps):

                offset = n_rnn_steps-step
                index = frame - offset

                if index <= 0:
                    continue

                if index%frequency == 0:

                    dpt[frame][step] = zeroing

                else:

                    dpt[frame][step] = zeroing
                    gnd[frame][step] = zeroing
                    obj[frame][step] = zeroing
                    bld[frame][step] = zeroing
                    veg[frame][step] = zeroing
                    sky[frame][step] = zeroing

    if option == 'rgbsd':

        for frame in range(0,n_frames):
            for step in range(0,n_rnn_steps):

                offset = n_rnn_steps-step
                index = frame - offset

                if index <= 0:
                    continue

                if index%frequency == 0:

                    dpt[frame][step] = zeroing
                    gnd[frame][step] = zeroing
                    obj[frame][step] = zeroing
                    bld[frame][step] = zeroing
                    veg[frame][step] = zeroing
                    sky[frame][step] = zeroing

                else:

                    dpt[frame][step] = zeroing
                    gnd[frame][step] = zeroing
                    obj[frame][step] = zeroing
                    bld[frame][step] = zeroing
                    veg[frame][step] = zeroing
                    sky[frame][step] = zeroing


    return [imr,img,imb,dpt,gnd,obj,bld,veg,sky]















