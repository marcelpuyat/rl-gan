# Adapted from https://www.hackerearth.com/practice/notes/beautiful-python-a-simple-ascii-art-generator-from-images/
import numpy as np

ASCII_CHARS = [ '.', '?', '%', '.', 'S', '+', '#', '*', ':', ',', '@']

def im_to_ascii(im, mn=None, mx=None):
    """
    Maps each pixel to an ascii char based on the range in which it lies.
    Intensities are divided into 11 ranges of 25 pixels each.

    im: 2D numpy ndarray representing grayscale image
    mn: hand-coded min intensity value
    mx: hand-coded max intensity value
    """
    if mn is None:
        mn = np.min(im)
    else:
        im = np.clip(im, mn, np.inf)
    if mx is None:
        mx = np.max(im)
    else:
        im = np.clip(im, -np.inf, mx)

    bin_width = int(np.ceil((mx - mn + 1.)/len(ASCII_CHARS)))
    im_binned = (im - mn)/bin_width
    
    return '\n'.join([''.join([ASCII_CHARS[val] for val in row]) for row in im_binned])

