import numpy as np
import tensorflow as tf
from config import *
from gym import Env, spaces
from gym.utils import seeding
from discriminator import *
from ascii_render import im_to_ascii

class DrawEnv(Env):
	def __init__(self, dimension=MNIST_DIMENSION, digit=None):
		# Dimensions of the drawing. Note that the drawing will always be
		# a square, so the dimension is both the height and the width.
		self.dimension = dimension

		# One action per possible pixel value
		self.action_space = spaces.Discrete(NUM_POSSIBLE_PIXEL_VALUES-1)

		# Values of the drawing's pixels
		self.observation_space = spaces.MultiDiscrete([NUM_POSSIBLE_PIXEL_VALUES, dimension*dimension])

		self._reset_pixel_values()

		self.last_real_prob = 0
		self.last_fake_prob = 0
		self.coordinate = 0

		# The actual set of numbers the agent is trying to draw.
		self.digits = MNIST_DIGITS
		self.one_class = (digit is not None)
		self.number = (digit if self.one_class else np.random.choice(self.digits))

		
	def set_session(self, sess):
		self.sess = sess
		train_class = (self.number if self.one_class else None)
		self.rl_discriminator = RLDiscriminator(self.sess, self.dimension, self.dimension,
												batch_size=1, train_class=train_class)

		
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
		if not self.one_class:
			self.number = np.random.choice(self.digits)

		return {'number': self.number, 'pixels': np.copy(self.pixel_values), 'coordinate': self.coordinate}


	# Fill a pixel.
	def step(self, a):
		assert a < MAX_PX_VALUE, "Pixel value for an action must fall in range: [0,{}], tried {}"\
			                     .format(MAX_PX_VALUE-1, a)
		assert a >= 0, "Pixel value for an action must fall in range: [0,{}], tried {}"\
  			           .format(MAX_PX_VALUE-1, a)

		# Set the pixel value in our state based on the action.
		self.pixel_values[self.coordinate] = a + MIN_PX_VALUE

		# print("Testing disc with taking real action " + str(a+2) + " at coordinate: " + str(self.coordinate))
		self.coordinate += 1

		done = (self.coordinate == self.dimension*self.dimension)
		num_unfilled = self.pixel_values.size - self.coordinate - 1
		if (self.last_fake_prob > 0.1 or self.last_real_prob < 0.9):
			#print("Actually training Disc")
			self.last_fake_prob, self.last_real_prob = self.rl_discriminator.train(self.pixel_values, self.number, num_unfilled, debug=False)
		else:
			#print("Not training Disc")
			self.last_fake_prob, self.last_real_prob = self.rl_discriminator.get_disc_loss(self.pixel_values, self.number, num_unfilled, debug=False)

		return ({'number': self.number,
				 'pixels': np.copy(self.pixel_values),
				 'coordinate': self.coordinate}, # Observation
				 self._compute_reward(self.last_fake_prob[0][0], 0), # Reward
				 done, # Whether done
				 {}) # Info

	
	# This is used to see what reward one would get by trying a given action from the current state
	# without actually taking that action.
	def try_step(self, a):
		assert a < MAX_PX_VALUE, "Pixel value for an action must fall in range: [0,{}], tried {}"\
			                     .format(MAX_PX_VALUE-1, a)
		assert a >= 0, "Pixel value for an action must fall in range: [0,{}], tried {}"\
  			           .format(MAX_PX_VALUE-1, a)
		
		copy = np.copy(self.pixel_values)
		copy[self.coordinate] = a + MIN_PX_VALUE

		num_unfilled = copy.size - self.coordinate - 1

		# print("Testing disc with taking try action " + str(a+2) + " at coordinate: " + str(self.coordinate))
		fake_prob, _ = self.rl_discriminator.get_disc_loss(copy, self.number, num_unfilled, debug=False)
		return None, self._compute_reward(fake_prob[0][0], 0), False, {} 

		
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


	### Discriminator functions ###
	def discrim_loss_tensors(self):
		return self.rl_discriminator.loss_tensors()

	
	def get_discrim_placeholders(self):
		return (self.rl_discriminator.get_real_placeholder(),
			    self.rl_discriminator.get_real_label_placeholder(),
			    self.rl_discriminator.get_fake_placeholder(),
			    self.rl_discriminator.get_fake_label_placeholder())

	
	def get_discrim_placeholder_values(self):
		real, real_labels = self.rl_discriminator.get_real_batch(0)
		fake, fake_labels = self.rl_discriminator.get_fake_batch(self.pixel_values, self.number, 0)
		return real, real_labels, fake, fake_labels

	
	def train_disc_random_fake(self):
		rand_fake = np.random.randint(MIN_PX_VALUE, 1+MAX_PX_VALUE, (self.dimension*self.dimension))
		num_unfilled = np.random.randint(0, self.dimension*self.dimension)
		if num_unfilled != 0:
			rand_fake[-num_unfilled:] = UNFILLED_PX_VALUE
		fake_prob, real_prob = self.rl_discriminator.train(rand_fake, self.number, num_unfilled, debug=False)
		return fake_prob, real_prob

	

########### ENV TRAINING DISC ON DEMAND AND ONLY REWARDING AT END OF EPISODES ##########
class DrawEnvTrainOnDemand(DrawEnv):
	def __init__(self, batch_size=10, dimension=MNIST_DIMENSION, digit=8):
		DrawEnv.__init__(self, dimension, digit)
		self.disc_batch_size = batch_size

		
	def set_session(self, sess):
		self.sess = sess
		train_class = (self.number if self.one_class else None)
		self.rl_discriminator = RLDiscriminatorFullImagesOnly(self.sess, self.dimension, self.dimension,
															  batch_size=self.disc_batch_size,
															  train_class=train_class)
		
			
	def step(self, a):
		assert a < MAX_PX_VALUE, "Pixel value for an action must fall in range: [0,{}], tried {}"\
			                     .format(MAX_PX_VALUE-1, a)
		assert a >= 0, "Pixel value for an action must fall in range: [0,{}], tried {}"\
  			           .format(MAX_PX_VALUE-1, a)

		# Set pixel value based on action.
		self.pixel_values[self.coordinate] = a + MIN_PX_VALUE
		self.coordinate += 1
		done = (self.coordinate == self.dimension*self.dimension)
		if done: # Evaluate discriminator on completed image
			fake_probs, real_probs = self.rl_discriminator.get_disc_loss(self.pixel_values, self.number)
			self.last_fake_prob = fake_probs[-1][0]
			self.last_real_prob = real_probs[-1][0]
		reward = (0 if not done else self._compute_reward(self.last_fake_prob))

		return ({'number': self.number,
				 'pixels': np.copy(self.pixel_values),
				 'coordinate': self.coordinate}, # Observation
				reward, done, {})

	
	def try_step(self, a):
		raise NotImplementedError('Not supported in DrawEnvTrainOnDemand')

	
	# Get negative of generator GAN loss from Goodfellow et al. 2014's GAN definition
	def _compute_reward(self, fake_prob):
		return -np.log(1 - fake_prob)


	### Discriminator functions ###

	# Train on a random fake example without setting values to UNFILLED_PX_VALUE
	def train_disc_random_fake(self):
		rand_fake = np.random.randint(MIN_PX_VALUE, 1+MAX_PX_VALUE,
									  size=(self.disc_batch_size, self.dimension*self.dimension))
		rand_labels = np.random.choice(self.digits, size=(self.disc_batch_size,1))
		fake_probs, real_probs = self.rl_discriminator.train(rand_fake, rand_labels)		
		return np.mean(fake_probs), np.mean(real_probs)

	
	# Train the discriminator using some fake images provided by the generator. 
	def train_discriminator(self, fake_images, fake_labels):
		assert self.disc_batch_size == len(fake_images) == len(fake_labels), 'Number of fake examples must be same as'\
			+ 'batch_size {}. Got len(fake_images) = {}, len(fake_labels) = {}'\
			.format(self.disc_batch_size, len(fake_images), len(fake_labels))
		
		if self.last_fake_prob <= 0.1 or self.last_real_prob >= 0.9: # don't train disc if too strong
			fake_probs, real_probs = self.rl_discriminator.get_disc_loss(fake_images, fake_labels)
			self.last_fake_prob = fake_probs[-1][0]
			self.last_real_prob = real_probs[-1][0]
		else:
			fake_probs, real_probs = self.rl_discriminator.train(fake_images, fake_labels)
			self.last_fake_prob = fake_probs[-1][0]
			self.last_real_prob = real_probs[-1][0]
