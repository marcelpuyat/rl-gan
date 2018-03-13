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
    x, y = index_to_xy(coord)
    window = im_pad[y-l_radius:y+r_radius+1, x-l_radius:x+r_radius+1]
    return window.flatten()
    
