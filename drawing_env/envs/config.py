import numpy as np
import mnist.mnist as mnist

MNIST_DIMENSION = 28
MNIST_DIGITS = range(10)

NUM_POSSIBLE_PIXEL_VALUES = 3
# Bin intensities 0-255 into NUM_POSSIBLE_PIXEL_VALUES-1 ranges.
# Add 1 to MAX_VAL so that if MAX_VAL divisible by NUM_POSSIBLE_PIXEL_VALUES-1,
# we still only have NUM_POSSIBLE_PIXEL_VALUES-1 buckets for filled pixels
BIN_WIDTH = int(np.ceil((mnist.MAX_VAL+1.)/(NUM_POSSIBLE_PIXEL_VALUES-1)))

UNFILLED_PX_VALUE = 0
MIN_PX_VALUE = 1 + UNFILLED_PX_VALUE
MAX_PX_VALUE = -1 + NUM_POSSIBLE_PIXEL_VALUES
