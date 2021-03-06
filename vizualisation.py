import cv2
import numpy as np

background_rgb = [150,160,150]
def append_frame(mat_list):
    if(not isinstance(mat_list,type(['list']))):
        mat_list = [mat_list]
    N_mat = len(mat_list)
    size_x = 0
    size_y = 0
    border_size = [10,10]
    for idx in range(N_mat):
        shp = mat_list[idx].shape
        print(mat_list[idx].dtype)
        if(shp[0] > size_y):
            size_y = shp[0]
        size_x = size_x + shp[1]

    size_x = size_x + (N_mat+1) * border_size[0]
    size_y = size_y + 2 * border_size[1]
    full_frame = np.zeros((size_y,size_x,3),dtype=np.uint8)
    full_frame[:,:] = background_rgb

    idx_start = border_size[1]
    jdx_start = border_size[0]
    for idx in range(N_mat):
        mat = mat_list[idx]
        shp = mat.shape
        full_frame[jdx_start:jdx_start+shp[0],idx_start:idx_start+shp[1],:] = mat
        idx_start += border_size[1] + shp[1]
        
    return full_frame

def merge_frame(mat_list,txt_list=None,option=None,window_size=[1000,1000],border_size=[10,10]):
    '''
    each subpicture will be displayed as square
    '''

    ### check that the input is a list of matrices

    if(not isinstance(mat_list,type(['list']))):
        mat_list = [mat_list]
    N_mat = len(mat_list)
    N_row = np.floor(np.sqrt(N_mat))
    
    if(isinstance(txt_list,type(None))):
        txt_list = [' '] * N_mat

    shp = [] # [width,height,N_filters]

    ### Get shape and dimensions info for each matrix
    N_square_cum = [0] * (N_mat+1)
    count = 0
    for idx in range(N_mat):
        shape_mat = mat_list[idx].shape
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
                t_N_square_per_column[idx_row] = t_N_square_max
            t_square_per_row[idx_row] += tmp

            if(N_square_cum[idx+1] >= (idx_row+1) * t_N_square_per_row):
                
                t_N_square_max = 0
                idx_row += 1
                
        
        t_square_size = (window_size[0]-100) // max(max(t_square_per_row),sum(t_N_square_per_column))
        if(t_square_size>square_size):
            square_size = int(t_square_size)
            N_square_per_column = t_N_square_per_column
            row = t_row
        else:
            break

    ### Defines size for borders between matrices (big_border), between filters (small_border)
    small_border = [1,1]
    big_border = [25,25]


    ### Dimension for each filter (excluding border)
    real_square_size = [square_size - small_border[0]] * 2

    ### Array initialization for location of first pixel for each matrix
    pixel_x = [big_border[0]] * N_mat
    pixel_y = [big_border[1]] * N_mat
    idx_col = big_border[1]
    for idx in range(1,N_mat):
        if(row[idx]==row[idx-1]):
            pixel_x[idx] = pixel_x[idx-1] + shp[idx-1][3] * square_size + big_border[0]
        else:
            idx_col += N_square_per_column[row[idx-1]] * square_size + big_border[1]
        pixel_y[idx] = int(idx_col) 


    ### Write the matrices into a single frame
    B = np.zeros((window_size[0],window_size[1],1))-1
    for idx in range(N_mat):
        mat = mat_list[idx]
        filter_N = 0
        N_square_x = shp[idx][3]
        N_square_total = shp[idx][2]
        start_idx = pixel_x[idx] 
        start_jdx = pixel_y[idx] 
        
        if(start_idx+N_square_x*square_size < window_size[0]):
            for square_idx in range(N_square_x):
                for square_jdx in range(N_square_x):
                    if(filter_N<N_square_total):
                        filter_ij = cv2.resize(mat[0,:,:,filter_N],tuple(real_square_size))
                        B[start_jdx:start_jdx + real_square_size[1],start_idx:start_idx + real_square_size[0],0] = filter_ij
                        filter_N += 1
                    start_jdx = start_jdx + square_size
                start_idx = start_idx + square_size
                start_jdx = pixel_y[idx] 


    ### Color tuning for the final output
    mask = np.copy(B)
    mask[mask>-1] = 0
    mask[mask==-1] = 255
    mask = mask.astype(np.uint8)
    C = (B*255).astype(np.uint8)
    
    C = cv2.applyColorMap(C, cv2.COLORMAP_JET)
    D = cv2.bitwise_and(C,C,mask=mask)


    ### Text added for each matrix
    for idx in range(N_mat): 
        cv2.putText
        C = cv2.subtract(C,D)
        D[:,:] = background_rgb
        D = cv2.bitwise_and(D,D,mask=mask)
        C = cv2.add(C,D)


    for idx in range(N_mat):
        cv2.putText(C, txt_list[idx], (pixel_x[idx],pixel_y[idx]-int(big_border[1]*0.2)), 
                0,  big_border[0]/50, [50,50,50],1)


    return C


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


def display(frame):
    while True:
        cv2.imshow('frame_{}'.format(1),frame)
        k = cv2.waitKey(30)
        # print(k)
        if k in [-1,255]:
            continue
        elif(k in [27,113,99]):
            exit("Interrupted by user")
        else: 
            break
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

    # cv2.waitKey();

def show():
    cv2.waitKey()

def main():
    print('Not implemented')

if __name__ == '__main__':
    main()