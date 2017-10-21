import numpy as np



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
        if inv_depth[i] > 0.02:
            depth[i] = 1./inv_depth[i]
    return depth


