import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.style as style
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



def plot_training_loss(losses,n_epochs,name,saving=True):

    plt.plot(losses)
    plt.title('Training Losses')
    plt.xlabel('# Epochs')
    plt.ylabel('Loss')
    plt.xlim([0,n_epochs])
    plt.ylim([0,losses[0]+20000])

    if saving == True:
        plot_name = 'training_losses'
        plot_folder = './plots/'
        plot_path = plot_folder + plot_name + '.png'
        plt.savefig(plot_path)

    plt.ion()
    plt.show()
    plt.pause(0.005)
    plt.close('all')


def print_training_frames(input_frame,output_frame,label_frame,shape,channel='all'):

    if channel=='red':
        im_shape = (shape[0],shape[1])

        im_r_input = np.reshape(input_frame['xcr1'],im_shape).T
        im_r_output = np.reshape(output_frame['xcr1'],im_shape).T
        im_r_label = np.reshape(label_frame['xcr1'],im_shape).T



        fig,axes = plt.subplots(1,3)

        axes[0].imshow(im_r_input)
        axes[0].set_title('Red Channel Input')

        axes[1].imshow(im_r_output)
        axes[1].set_title('Red Channel Output')

        axes[2].imshow(im_r_label)
        axes[2].set_title('Red Channel Label')

        plt.ion()
        plt.show()
        plt.pause(0.0001)
        plt.close('all')







