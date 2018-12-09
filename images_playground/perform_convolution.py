import numpy as np 
from math import floor, ceil
from scipy.signal import convolve

def pad_zeros(matrix, pad_dim):
    """
    pad_dim needs to be a sequence of two length. 
    
    """
    
    existing_dim =  matrix.shape
    
    new_dim = (pad_dim[0]*2 + existing_dim[0], pad_dim[1]*2 + existing_dim[1])
    
    new_matrix = np.zeros(new_dim)
    
    new_matrix[pad_dim[0]: pad_dim[0]+ existing_dim[0], pad_dim[1]: pad_dim[1]+ existing_dim[1]] = matrix
    
    return new_matrix

def weighted_sum(matrix_one, matrix_two):
    sum = 0
    
    for i, j in enumerate(matrix_one):    
        for a, b in zip(j, matrix_two[i]):
            sum = sum + a*b
            
    return sum

def flip_vertically(matrix):
    return matrix[::-1]

def flip_horizontally(matrix):
    return matrix[:, ::-1]

def convolution(image, kernel, stride = 1):
    
    w = flip_vertically(flip_horizontally(kernel))

    dim_image = image.shape
    dim_kernel = w.shape
    stride = 1

    dim_kernel_center = (floor((dim_kernel[0]- 1 )/2),floor((dim_kernel[1]- 1 )/2))

    padding_dim = dim_kernel_center

    dim_output_matrix = (floor((dim_image[0] + 2* padding_dim[0] - dim_kernel[0])/stride) +1, \
                            floor((dim_image[1] + 2* padding_dim[1] - dim_kernel[1])/stride)+1)


    output_matrix = np.zeros(dim_output_matrix) 
    padded_matrix = pad_zeros(image, padding_dim)

    rstep = 0
    for r in range(dim_output_matrix[0]):

        step = 0
        for c in range(dim_output_matrix[1]):

            output_matrix[r,c] = weighted_sum(padded_matrix[ rstep:dim_kernel[0]+ rstep , step : dim_kernel[1] + step ], w)
            step = step+stride

        rstep = rstep + stride

    print(output_matrix)
    return output_matrix

m = np.array([[1,1,1],[2,2,2],[3,3,3]])
w = np.array([[-1,2,-1],[-1,2,-1],[-1,2,-1]])

convolution(m, w)


m = np.array([[1,3,2,4],[5,6,1,3],[1,2,0,2],[3,4,3,2]])
w = np.array([[1,0,3],[1,2,1],[0,1,1]])

convolution(m, w)