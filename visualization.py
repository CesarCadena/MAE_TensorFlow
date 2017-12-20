import numpy as np
import matplotlib.pyplot as plt


from copy import copy

def display_sequence(seq,shape):

    # display rgb sequence:


    im_shape = (shape[1],shape[0])


    iterator = 0

    for frame in seq:

        im_r = np.reshape(frame['xcr1'],im_shape).T
        im_g = np.reshape(frame['xcg1'],im_shape).T
        im_b = np.reshape(frame['xcb1'],im_shape).T

        im_rgb = np.dstack((im_r,im_g,im_b))


        plt.imshow(im_rgb)
        plt.savefig('plots/Images/im_rgb_'+str(iterator) + '.png')

        im_depth = frame['xid1']

        for i in range(0,im_shape[0]*im_shape[1]):

            if im_depth[i] == -1:
                im_depth[i] = 0

        im_depth = np.reshape(im_depth,im_shape).T

        plt.imshow(im_depth,cmap='gist_ncar')
        plt.savefig('plots/Images/im_dpt_'+str(iterator)+'.png')

        im_sem = np.reshape(frame['sem1'],im_shape).T

        plt.imshow(im_sem,cmap='Dark2')
        plt.savefig('plots/Images/im_sem_'+str(iterator)+'.png')



        iterator += 1




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

    im_depth = frame['xid1']
    for i in range(0,im_shape[0]*im_shape[1]):
        if im_depth[i] == -1:
            im_depth[i] = 0

    im_depth = np.reshape(im_depth,im_shape).T

    im_sem = np.reshape(frame['sem1'],im_shape).T

    plt.imshow(im_rgb)
    plt.show()

    plt.imshow(im_depth,cmap='gist_ncar')
    plt.show()

    plt.imshow(im_sem,cmap='Dark2')
    plt.show()


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


def print_training_frames(input_frame,output_frame,label_frame,shape,channel='all',savefig=False,i=0):

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

        if savefig==True:
            plt.savefig('plots/fig_val' + str(i) +'.png')

        plt.ion()
        plt.show()
        plt.pause(0.0001)
        plt.close('all')



def print_validation_frames(input_frame,output_frame,label_frame,shape,channel='all'):

    if channel=='red':
        im_shape = (shape[0],shape[1])


        im_r_input = np.reshape(input_frame,im_shape).T
        im_r_output = np.reshape(output_frame,im_shape).T
        im_r_label = np.reshape(label_frame,im_shape).T




        fig,axes = plt.subplots(1,3)

        axes[0].imshow(im_r_input)
        axes[0].set_title('Red Channel Input')

        axes[1].imshow(im_r_output)
        axes[1].set_title('Red Channel Output')

        axes[2].imshow(im_r_label)
        axes[2].set_title('Red Channel Label')

        plt.savefig()

        plt.ion()
        plt.show()
        plt.pause(0.0001)
        plt.close('all')

    if channel=='depth':
        im_shape = (shape[0],shape[1])

        input_frame = np.asarray(input_frame)

        label_frame = np.asarray(label_frame)
        label_frame = np.reshape(label_frame,(1,shape[0]*shape[1]))


        for i in range(0,input_frame.shape[1]):
            if input_frame[0,i] == -1.0:
                input_frame[0,i] = 0
            if label_frame[0,i] == -1.0:
                label_frame[0,i] = 0

        im_depth_input = np.reshape(input_frame,im_shape).T
        im_depth_output = np.reshape(output_frame,im_shape).T
        im_depth_label = np.reshape(label_frame,im_shape).T

        fig,axes = plt.subplots(1,3)

        axes[0].imshow(im_depth_input,cmap='hsv')
        axes[0].set_title('Depth Channel Input')

        axes[1].imshow(im_depth_output,cmap='hsv')
        axes[1].set_title('Depth Channel Output')

        axes[2].imshow(im_depth_label,cmap='hsv')
        axes[2].set_title('Depth Channel Label')


        plt.ion()
        plt.show()
        plt.pause(0.0001)
        plt.close('all')


def display_sequence_mirroring(imr,imb,img,dpt,gnd,obj,bld,veg,sky,imr_m,imb_m,img_m,dpt_m,gnd_m,obj_m,bld_m,veg_m,sky_m):

    n_steps = len(imr)

    shape = (60,18)

    rgb = []
    rgb_m = []

    for step in range(0,n_steps):
        rgb.append(np.dstack((np.reshape(imr[step],shape).T,np.reshape(img[step],shape).T,np.reshape(imb[step],shape).T)))
        rgb_m.append(np.dstack((np.reshape(imr_m[step],shape).T,np.reshape(img_m[step],shape).T,np.reshape(imb_m[step],shape).T)))


    fig,axes = plt.subplots(2,n_steps)

    for step in range(0,n_steps):
        axes[0,step].imshow(rgb[step])
        axes[1,step].imshow(rgb_m[step])

    plt.show()

    fig,axes = plt.subplots(2,n_steps)

    for i in range(0,n_steps):
        for j in range(0,1080):
            if dpt[i][j] == -1:
                dpt[i][j] = 0
            if dpt_m[i][j] == -1:
                dpt_m[i][j] = 0

    for step in range(0,n_steps):
        axes[0,step].imshow(np.reshape(dpt[step],shape).T,cmap='gist_ncar')
        axes[1,step].imshow(np.reshape(dpt_m[step],shape).T,cmap='gist_ncar')

    plt.show()

    fig,axes = plt.subplots(2,n_steps)

    for step in range(0,n_steps):

        frame = np.zeros((1,1080)).astype(int)
        frame_m = copy(frame)

        for i in range(0,1080):
            if gnd[step][i] == 1:
                frame[0,i] = 1

            if obj[step][i] == 1:
                frame[0,i] = 2

            if bld[step][i] == 1:
                frame[0,i] = 3

            if veg[step][i] == 1:
                frame[0,i] = 4

            if sky[step][i] == 1:
                frame[0,i] = 5

            if gnd_m[step][i] == 1:
                frame_m[0,i] = 1

            if obj_m[step][i] == 1:
                frame_m[0,i] = 2

            if bld_m[step][i] == 1:
                frame_m[0,i] = 3

            if veg_m[step][i] == 1:
                frame_m[0,i] = 4

            if sky_m[step][i] == 1:
                frame_m[0,i] = 5


        axes[0,step].imshow(np.reshape(frame,shape).T,cmap='Dark2')
        axes[1,step].imshow(np.reshape(frame_m,shape).T,cmap='Dark2')

    plt.show()







#def full_model_visualiuzation(input_frame,output_frame,label_frame,shape,resolution=(18,60)):







