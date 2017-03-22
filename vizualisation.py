import cv2
import numpy as np


def merge_frame(mat_list,txt_list=None,option=None,window_size=[1000,1000],border_size=[10,10]):
    '''
    this function is designed to show result of various filters from differnt convolutional layers
    '''


    if(not isinstance(mat_list,type(['list']))):
        mat_list = [mat_list]
    N_mat = len(mat_list)
    N_mat_per_row = 6
    N_row = np.floor(np.sqrt(N_mat))
    
    if(isinstance(txt_list,type(None))):
        txt_list = [' '] * N_mat

def convolution(mat,window_size=[1000,1000],border_size=[10,10],frame_index=0):
    shp = mat.shape

    B = np.zeros((window_size[0],window_size[1],1))
    N = int(np.ceil(np.sqrt(shp[2])))
    size = [int(x) for x in [np.floor(window_size[0]/N),np.floor(window_size[1]/N)]]
    resize_size = [size[0]-border_size[0],size[1]-border_size[1]]

    
    



    
    N_image = 0
    start_idx = 0
    for idx in range(N):
        start_jdx = 0
        for jdx in range(N):
            
            if(N_image < shp[2]):
                filter_ij = cv2.resize(mat[:,:,N_image],tuple(resize_size))
                B[start_idx:start_idx + resize_size[0],start_jdx:start_jdx + resize_size[1],0] = filter_ij
            N_image += 1
            start_jdx += size[1]
        start_idx += size[0]
    if(frame_index>-1):
        cv2.imshow('d{}'.format(frame_index),B)
    else:
        cv2.imshow('d0',B)
        cv2.waitKey();

def show():
    cv2.waitKey()