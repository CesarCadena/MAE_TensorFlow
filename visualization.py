import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf




def display_frame(frame,shape):
    '''

    :param frame: dict with all different modalities used in the MAE
    :return:
    '''

    im_shape = (shape[1],shape[0])

    im_r = np.reshape(frame['xcr1'],im_shape).T
    im_g = np.reshape(frame['xcg1'],im_shape).T
    im_b = np.reshape(frame['xcb1'],im_shape).T

    im_rgb = np.dstack((im_r,im_g,im_b))
    im_depth = np.reshape(frame['xid1'],im_shape).T

    im_sem = np.reshape(frame['sem1'],im_shape).T


    fig,axes = plt.subplots(1,3)

    axes[0].imshow(im_rgb)
    axes[0].set_title('RGB Image')

    axes[1].imshow(im_depth,cmap='gist_ncar')
    axes[1].set_title('Inv Depth Image')

    axes[2].imshow(im_sem,cmap='Dark2')
    axes[2].set_title('Image Semantics')

    plt.show()
    plt.close()

    return 0
