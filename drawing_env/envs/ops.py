import tensorflow as tf
import numpy as np

def conv2d(input_, output_dim, kernel_h=5, kernel_w=5, stride_h=2, stride_w=2, stddev=0.02, name="conv2d"):
	with tf.variable_scope(name):
		w = tf.get_variable('w', [kernel_h, kernel_w, input_.get_shape()[-1], output_dim],
			initializer=tf.truncated_normal_initializer(stddev=stddev))
		conv = tf.nn.conv2d(input_, w, strides=[1, stride_h, stride_w, 1], padding='SAME')

		biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
		conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
		tf.summary.histogram("weights", w)
		return conv

def lrelu(x, leak=0.2, name="lrelu"):
	return tf.maximum(x, leak*x)

def dense(input, output_dim, stddev=0.02, name='dense'):
	with tf.variable_scope(name):
		shape = input.get_shape()
		W = tf.get_variable('denseW', [shape[1], output_dim],
						initializer=tf.random_normal_initializer(stddev=stddev))
		b = tf.get_variable('denseb', [output_dim],
							initializer=tf.zeros_initializer())
		tf.summary.histogram("weights", W)
		return tf.matmul(input, W) + b

def gaussian_noise_layer(input, noise_std=0.2):
	# TODO: Convert this to an actual noise layer instead of randomly flipping bits.
	return tf.clip_by_value(np.random.choice([2,0.5,1], input.shape, p=[0.2, 0.2, 0.6]) * input, 1, 2)