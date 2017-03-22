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

    big_border = [20,20]
    small_border = [5,5]
    
    shp = [] # [width,height,N_filters]
    N_square_0 = [0] * N_mat
    N_square_cum = [0] * (N_mat+1)
    count = 0
    for idx in range(N_mat):
        shape_mat = mat_list[idx].shape
        N_square_0[idx//N_mat_per_row] += np.ceil(np.sqrt(shape_mat[3]))
        N_square_cum[idx+1] = N_square_cum[idx] + np.ceil(np.sqrt(shape_mat[3]))
        shp.append([shape_mat[1],shape_mat[2],shape_mat[3],int(np.ceil(np.sqrt(shape_mat[3])))])
    

    square_size = 0
    ## Find the best configuration for organizing the layers into a single frame
    ## maximize the size of the filters
    for jdx in range(10):
        N_row = jdx + 1
        t_row = [0] * N_mat
        t_square_per_row = [0] * N_mat
        idx_row = 0
        t_N_square_per_row = np.ceil(N_square_cum[-1] / N_row)
        t_N_square_per_column = [0] * N_mat
        t_N_square_max = 0
        for idx in range(N_mat):
            t_row[idx] = idx_row
            tmp = shp[idx][3]
            if(t_N_square_max<tmp):
                t_N_square_max = tmp
            t_square_per_row[idx_row] += tmp

            if(N_square_cum[idx+1] >= (idx_row+1) * t_N_square_per_row):
                t_N_square_per_column[idx_row] += t_N_square_max
                t_N_square_max = 0
                idx_row += 1
                
        # N_square_per_column += N_square_max
        
        t_square_size = (window_size[0]-100) // max(max(t_square_per_row),sum(t_N_square_per_column))
        if(t_square_size>square_size):
            square_size = int(t_square_size)
            N_square_per_column = t_N_square_per_column
            row = t_row
        else:
            break

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