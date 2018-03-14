from config import *
import numpy as np

# Convert vector index to image coordinates (x,y)
def index_to_xy(idx, dim=FULL_DIMENSION):
    return (int(idx) % dim, int(idx)/dim)

# Convert image vector to a FULL_DIMENSION x FULL_DIMENSION matrix.
def vec_to_im(vec):
    return vec.reshape((FULL_DIMENSION, FULL_DIMENSION))

# Return the values of a window_size * window_size area around the given
# coordinate. If window_size is even, coord is the top-left pixel of the
# 2x2 square at the center of the window
def get_local_pixels(all_pixels, coord, window_size=LOCAL_DIMENSION,
                     padding_style='reflect', **kwargs):
    radius_l = (window_size-1)/2
    radius_u = window_size/2
    im_pad = np.pad(vec_to_im(all_pixels), pad_width=radius_u, mode=padding_style, **kwargs)
#    print(all_pixels)
#    print('')
#    print(vec_to_im(all_pixels))
#    print('')
#    print(im_pad)
#    print('')
    x, y = index_to_xy(coord)
    x += radius_l; y += radius_l # to access (i,j) of original image, add radius_l to coords in im_pad
 #   print(x)
 #   print(y)
 #   print(radius_l)
 #   print(radius_u)
    window = im_pad[y-radius_l:y+radius_u+1, x-radius_l:x+radius_u+1]
 #   print(window)
 #   print('')
    return window.flatten()
    
