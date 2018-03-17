import numpy as np
import tensorflow as tf
from config import *
from gym import Env, spaces
from gym.utils import seeding
from discriminator import RLDiscriminator
from ascii_render import im_to_ascii

class DrawEnv(Env):
	def __init__(self, dimension=MNIST_DIMENSION):
		# Dimensions of the drawing. Note that the drawing will always be
		# a square, so the dimension is both the height and the width.
		self.dimension = dimension

		# MultiBinary gives us a zero or one action for each of (len)
		# passed in as an argument. So we pass in the total number of pixels,
		# dimension squared, to get the total action space.
		#
		# Note that the actions are to select 0 (meaning color white) or 1
		# (meaning color black) for a given pixel.
		self.action_space = spaces.Discrete(NUM_POSSIBLE_PIXEL_VALUES-1)

		# The first value in the action tuple is the pixel.
		#
		# The second value in the action is the pixel coordinate, flattened
		# into a 1d value.
		self.observation_space = spaces.MultiDiscrete([NUM_POSSIBLE_PIXEL_VALUES, dimension*dimension])

		self._reset_pixel_values()

		self.last_real_prob = 0
		self.last_fake_prob = 0
		self.coordinate = 0

		# The actual number the agent is trying to draw.
                self.digits = MNIST_DIGITS
                self.number = np.random.choice(self.digits)

                
	def set_session(self, sess):
		self.sess = sess
		self.rl_discriminator = RLDiscriminator(self.sess, self.dimension, self.dimension,
                                                        batch_size=1, train_class=self.number)

                
	def render(self, mode='human'):
		# TODO: Write this out to an actual file, and convert the pixel values to a format
		# that will allow the file to render as an actual image.

		# Right now, we're simply printing out the pixel_values to stdout.
		self._print_pixels(self.pixel_values)

                                
	def _print_pixels(self, pixels):
		print("-------------------------------------------")
                print(im_to_ascii(pixels.reshape((MNIST_DIMENSION,MNIST_DIMENSION))))
		print("\n")
                print("Number: " + str(self.number))
		print("-------------------------------------------")

                
	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random()
		return [seed]

        
	# Note this does not reset the discriminator parameters.
	def reset(self):
		self._reset_pixel_values()
		self.coordinate = 0
		self.number = np.random.choice(self.digits)

		return {'number': self.number, 'pixels': np.copy(self.pixel_values), 'coordinate': self.coordinate}

        
        # Reset, but deterministically choose the digit we wish to generate
        def reset_to_selected_digit(self, digit):
                self.reset()
                assert digit in self.digits, 'Selected digit must fall in range [{}, {}], inclusive'\
                                             .format(min(self.digits), max(self.digits))
                self.number = digit
                return {'number': self.number, 'pixels': np.copy(self.pixel_values), 'coordinate': self.coordinate}

        
	def discrim_loss_tensors(self):
		return self.rl_discriminator.loss_tensors()

        
	def train_disc_random_fake(self):
		rand_fake = np.random.randint(MIN_PX_VALUE, 1+MAX_PX_VALUE,
                                              (self.dimension*self.dimension))
		num_unfilled = np.random.randint(0, self.dimension*self.dimension)
		if num_unfilled != 0:
			rand_fake[-num_unfilled:] = UNFILLED_PX_VALUE
		fake_prob, real_prob = self.rl_discriminator.train(rand_fake, np.random.choice(self.digits),
                                                                   num_unfilled, debug=False)
		return fake_prob, real_prob

        
	# This is used to see what reward one would get by trying a given action from the current state
	# without actually taking that action.
	def try_step(self, a):
		assert a < NUM_POSSIBLE_PIXEL_VALUES-1, "Pixel value for an action must fall in range: [0," + str(NUM_POSSIBLE_PIXEL_VALUES-2) + "]. Current invalid action: " + str(a)
		assert a >= 0, "Pixel value for an action must fall in range: [0," + str(NUM_POSSIBLE_PIXEL_VALUES-2) + "]. Current invalid action: " + str(a)

		copy = np.copy(self.pixel_values)
		copy[self.coordinate] = a + MIN_PX_VALUE

		num_unfilled = copy.size - self.coordinate - 1

#		print("Testing disc with taking try action " + str(a+2) + " at coordinate: " + str(self.coordinate))
		fake_prob, _ = self.rl_discriminator.get_disc_loss(copy, self.number, num_unfilled, debug=False)
		return None, self._compute_reward(fake_prob[0][0], 0), False, {} 

        
	def get_discrim_placeholders(self):
		return (self.rl_discriminator.get_real_placeholder(),
                        self.rl_discriminator.get_real_label_placeholder(),
                        self.rl_discriminator.get_fake_placeholder(),
                        self.rl_discriminator.get_fake_label_placeholder())

        
	def get_discrim_placeholder_values(self):
		real, real_labels = self.rl_discriminator.get_real_batch(0)
		fake, fake_labels = self.rl_discriminator.get_fake_batch(self.pixel_values, self.number, 0)
		return real, real_labels, fake, fake_labels

        
	def step(self, a):
		assert a < NUM_POSSIBLE_PIXEL_VALUES-1, "Pixel value for an action must fall in range: [0," + str(NUM_POSSIBLE_PIXEL_VALUES-2) + "]. Current invalid action: " + str(a)
		assert a >= 0, "Pixel value for an action must fall in range: [0," + str(NUM_POSSIBLE_PIXEL_VALUES-2) + "]. Current invalid action: " + str(a)

		# Set the pixel value in our state based on the action.
		self.pixel_values[self.coordinate] = a + MIN_PX_VALUE

#		print("Testing disc with taking real action " + str(a+2) + " at coordinate: " + str(self.coordinate))
		self.coordinate += 1

		done = (self.coordinate == self.dimension*self.dimension)
		num_unfilled = self.pixel_values.size - self.coordinate - 1
		if (self.last_fake_prob > 0.1 or self.last_real_prob < 0.9):
			#print("Actually training Disc")
			self.last_fake_prob, self.last_real_prob = self.rl_discriminator.train(self.pixel_values, self.number, num_unfilled, debug=False)
		else:
			#print("Not training Disc")
			self.last_fake_prob, self.last_real_prob = self.rl_discriminator.get_disc_loss(self.pixel_values, self.number, num_unfilled, debug=False)

		return {'number': self.number, 'pixels': np.copy(self.pixel_values), 'coordinate': self.coordinate}, self._compute_reward(self.last_fake_prob[0][0], 0), done, {}

        
	def _reset_pixel_values(self):
		# The actual pixel values of the drawing. We start out with all values
		# equal to zero, meaning none of the pixel colors have been selected yet.
		self.pixel_values = np.full(self.dimension*self.dimension, UNFILLED_PX_VALUE)

                
	# The reward is a function of how much we were able to trick the discriminator (i.e. how
	# high the fake_prob is) and how many pixels had to be filled in.
	def _compute_reward(self, fake_prob, num_unfilled_pixels):
		# - We want the fake_prob to be correlated with the reward
		# - We want the num_unfilled_pixels to be inversely weighted with the reward
                return -np.log(1 - fake_prob)/(1 + num_unfilled_pixels)
