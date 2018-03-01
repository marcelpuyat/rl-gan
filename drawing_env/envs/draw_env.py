import numpy as np
import tensorflow as tf

from gym import Env, spaces
from gym.utils import seeding
from discriminator import RLDiscriminator

# Zero means the pixel color hasn't been selected yet.
# One means the pixel color is white.
# Two means the pixel color is black.
NUM_POSSIBLE_PIXEL_VALUES = 3

REWARD_FACTOR = 2

class DrawEnv(Env):
	def __init__(self, dimension=6):
		# Dimensions of the drawing. Note that the drawing will always be
		# a square, so the dimension is both the height and the width.
		self.dimension = dimension

		# MultiBinary gives us a zero or one action for each of (len)
		# passed in as an argument. So we pass in the total number of pixels,
		# dimension squared, to get the total action space.
		#
		# Note that the actions are to select 0 (meaning color white) or 1
		# (meaning color black) for a given pixel.
		self.action_space = spaces.MultiBinary(dimension*dimension*2)

		# The first value in the action tuple is the pixel.
		#
		# The second value in the action is the pixel coordinate, flattened
		# into a 1d value.
		self.observation_space = spaces.MultiDiscrete([NUM_POSSIBLE_PIXEL_VALUES, dimension*dimension])

		self._reset_pixel_values()

		self.sess = tf.Session()
		self.rl_discriminator = RLDiscriminator(self.sess, dimension, dimension, 1)
		self.last_real_prob = 0
		self.last_fake_prob = 0
		self.coordinate = 0

	def render(self, mode='human'):
		# TODO: Write this out to an actual file, and convert the pixel values to a format
		# that will allow the file to render as an actual image.

		# Right now, we're simply printing out the pixel_values to stdout.
		self._print_pixels(self.pixel_values)

	def render_val(self, val):
		if val == 2:
			return "-"
		if val == 3:
			return "*"

	def _print_pixels(self, pixels):
		print("--------------------------")
		print("\n")
		for row in xrange(self.dimension):
			row_str = ''
			for col in xrange(self.dimension):
				row_str += str(self.render_val(pixels[(row * self.dimension) + col])) + " "
			print(row_str + "\n")
		print("--------------------------")

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random()
		return [seed]

	# Note this does not reset the discriminator parameters.
	def reset(self):
		self._reset_pixel_values()
		self.coordinate = 0
		return {'pixels': np.copy(self.pixel_values), 'coordinate': self.coordinate}

	def get_reward_for_action_batch(self, pixel_values, actions, fill_policy=None):
		for a in actions:
			assert a[0] < NUM_POSSIBLE_PIXEL_VALUES, "Pixel value for an action must fall in range: [0," + str(NUM_POSSIBLE_PIXEL_VALUES-1) + "]. Current invalid action: " + str(a)
			assert a[0] >= 0, "Pixel value for an action must fall in range: [0," + str(NUM_POSSIBLE_PIXEL_VALUES-1) + "]. Current invalid action: " + str(a)
			assert a[1] < self.dimension * self.dimension, "Pixel coordinate for an action must fall in range [0," + str(self.dimension*self.dimension-1) + "]. Current invalid action: " + str(a)
			assert a[1] >= 0, "Pixel coordinate for an action must fall in range [0," + str(self.dimension*self.dimension-1) + "]. Current invalid action: " + str(a)

		state_batch = np.zeros((len(actions), self.dimension*self.dimension))
		next_state_batch = np.zeros((len(actions), self.dimension*self.dimension))
		num_unfilled_pixels = 0
		for idx, a in enumerate(actions):
			copied_pixels = np.copy(pixel_values)
			copied_pixels[a[1]] = a[0]
			next_state_batch[idx] = np.copy(copied_pixels)

			state_batch[idx], num_unfilled_pixels = self._fill_remaining_pixels(copied_pixels, fill_policy)

		probs_fake = self.rl_discriminator.get_disc_loss_batch(state_batch, False)
		rewards = np.zeros(len(probs_fake[0]))
		for idx, p in enumerate(probs_fake[0]):
			rewards[idx] = self._compute_reward(p, num_unfilled_pixels)

		return next_state_batch, rewards, num_unfilled_pixels == 0, {}

	def train_disc_random_fake(self):
		rand_fake = np.random.randint(2,4,(self.dimension*self.dimension))
		num_zeroed_out = np.random.randint(0, self.dimension*self.dimension)
		if num_zeroed_out != 0:
			rand_fake[-num_zeroed_out:] = 1
		fake_prob, real_prob = self.rl_discriminator.train(rand_fake, num_zeroed_out, True)
		return fake_prob, real_prob

	def _convert_action_idx_to_action(self, action_idx):
		# Remember action_idx is in the 1d flattened range of num_pixels * num_actions_per_pixel so
		# we have to do divison and modulus to fetch out the actual best pixel color and coordinate.
		return int((action_idx % 2) + 1), int(action_idx / 2)

	def try_step(self, a):
		assert a < NUM_POSSIBLE_PIXEL_VALUES, "Pixel value for an action must fall in range: [0," + str(NUM_POSSIBLE_PIXEL_VALUES-1) + "]. Current invalid action: " + str(a)
		assert a >= 0, "Pixel value for an action must fall in range: [0," + str(NUM_POSSIBLE_PIXEL_VALUES-1) + "]. Current invalid action: " + str(a)

		copy = np.copy(self.pixel_values)
		copy[self.coordinate] = a + 2

		num_zeroed_out = copy.size - self.coordinate

		print("Testing disc with taking try action " + str(a+2) + " at coordinate: " + str(self.coordinate))
		fake_prob, _ = self.rl_discriminator.get_disc_loss(copy, num_zeroed_out, True)
		return None, self._compute_reward(fake_prob[0][0], 0), False, {} 


	def step(self, a):
		assert a < NUM_POSSIBLE_PIXEL_VALUES, "Pixel value for an action must fall in range: [0," + str(NUM_POSSIBLE_PIXEL_VALUES-1) + "]. Current invalid action: " + str(a)
		assert a >= 0, "Pixel value for an action must fall in range: [0," + str(NUM_POSSIBLE_PIXEL_VALUES-1) + "]. Current invalid action: " + str(a)

		# Set the pixel value in our state based on the action.
		self.pixel_values[self.coordinate] = a + 2

		print("Testing disc with taking real action " + str(a+2) + " at coordinate: " + str(self.coordinate))
		self.coordinate += 1

		done = self.coordinate == self.dimension*self.dimension
		num_zeroed_out = self.pixel_values.size - self.coordinate
		if (self.last_fake_prob > 0.1 or self.last_real_prob < 0.9):
			print("Actually training Disc")
			self.last_fake_prob, self.last_real_prob = self.rl_discriminator.train(self.pixel_values, num_zeroed_out, True)
		else:
			print("Not training Disc")
			self.last_fake_prob, self.last_real_prob = self.rl_discriminator.get_disc_loss(self.pixel_values, num_zeroed_out, True)

		return {'pixels': np.copy(self.pixel_values), 'coordinate': self.coordinate}, self._compute_reward(self.last_fake_prob[0][0], 0), done, {}

	# def step(self, action_idx):
	# 	a = self._convert_action_idx_to_action(action_idx)
	# 	# First check if action is valid.
	# 	assert a[0] < NUM_POSSIBLE_PIXEL_VALUES, "Pixel value for an action must fall in range: [0," + str(NUM_POSSIBLE_PIXEL_VALUES-1) + "]. Current invalid action: " + str(a)
	# 	assert a[0] >= 0, "Pixel value for an action must fall in range: [0," + str(NUM_POSSIBLE_PIXEL_VALUES-1) + "]. Current invalid action: " + str(a)
	# 	assert a[1] < self.dimension * self.dimension, "Pixel coordinate for an action must fall in range [0," + str(self.dimension*self.dimension-1) + "]. Current invalid action: " + str(a)
	# 	assert a[1] >= 0, "Pixel coordinate for an action must fall in range [0," + str(self.dimension*self.dimension-1) + "]. Current invalid action: " + str(a)

	# 	# If the action is None or if it sets a pixel to 0 or if this coordinate is already set.
	# 	if a is None or a[0] == 0 or self.pixel_values[a[1]] != 0:
	# 		if self.pixel_values[a[1]] != 0:
	# 			print("Trying to set pixel coordinate " + str(a[1]) + " to " + str(a[0]) + " when a value is already selected for it: " + str(self.pixel_values[a[1]]))
	# 		return np.copy(self.pixel_values), -100, False, {}

	# 	# Set the pixel value in our state based on the action.
	# 	self.pixel_values[a[1]] = a[0]

	# 	# We can do this because np.all returns true unless there is any zero-valued pixel (meaning
	# 	# a pixel color hasn't been selected for one coordinate).
	# 	done = np.all(self.pixel_values)

	# 	if not done:
	# 		return np.copy(self.pixel_values), 0, False, {}

	# 	if (self.last_fake_prob > 0.25 or self.last_real_prob < 0.75):
	# 		print("Actually training Disc")
	# 		self.last_fake_prob, self.last_real_prob = self.rl_discriminator.train(self.pixel_values, True)
	# 	else:
	# 		print("Not training Disc")
	# 		self.last_fake_prob, self.last_real_prob = self.rl_discriminator.get_disc_loss(self.pixel_values, True)

	# 	return np.copy(self.pixel_values), self._compute_reward(self.last_fake_prob[0][0], 0), done, {}


	def step_with_fill_policy(self, action_idx, fill_policy=None):
		a = self._convert_action_idx_to_action(action_idx)
		# First check if action is valid.
		assert a[0] < NUM_POSSIBLE_PIXEL_VALUES, "Pixel value for an action must fall in range: [0," + str(NUM_POSSIBLE_PIXEL_VALUES-1) + "]. Current invalid action: " + str(a)
		assert a[0] >= 0, "Pixel value for an action must fall in range: [0," + str(NUM_POSSIBLE_PIXEL_VALUES-1) + "]. Current invalid action: " + str(a)
		assert a[1] < self.dimension * self.dimension, "Pixel coordinate for an action must fall in range [0," + str(self.dimension*self.dimension-1) + "]. Current invalid action: " + str(a)
		assert a[1] >= 0, "Pixel coordinate for an action must fall in range [0," + str(self.dimension*self.dimension-1) + "]. Current invalid action: " + str(a)

		# If the action is None or if it sets a pixel to 0 or if this coordinate is already set.
		if a is None or a[0] == 0 or self.pixel_values[a[1]] != 0:
			if self.pixel_values[a[1]] != 0:
				print("Trying to set pixel coordinate " + str(a[1]) + " to " + str(a[0]) + " when a value is already selected for it: " + str(self.pixel_values[a[1]]))
			return self.pixel_values, -10, False, {}

		# Set the pixel value in our state based on the action.
		self.pixel_values[a[1]] = a[0]

		# We can do this because np.all returns true unless there is any zero-valued pixel (meaning
		# a pixel color hasn't been selected for one coordinate).
		done = np.all(self.pixel_values)

		fake_image, num_unfilled_pixels, tried_to_fill_already_filled_pixel = self._fill_remaining_pixels(self.pixel_values, fill_policy, True)
		print("Had to fill in " + str(num_unfilled_pixels) + " pixels.")
		if tried_to_fill_already_filled_pixel:
			print("During fill policy, tried to fill already filled pixel. Returning bad reward.")
			return self.pixel_values, -10 / num_unfilled_pixels, done, {}

		# Training the disc every iter for now...
		if (self.last_fake_prob > 0.25 or self.last_real_prob < 0.75):
			print("Actually training Disc")
			self.last_fake_prob, self.last_real_prob = self.rl_discriminator.train(fake_image, True)
		else:
			print("Not training Disc")
			self.last_fake_prob, self.last_real_prob = self.rl_discriminator.get_disc_loss(fake_image, True)

		return self.pixel_values, self._compute_reward(self.last_fake_prob, num_unfilled_pixels), done, {}

	def _reset_pixel_values(self):
		# The actual pixel values of the drawing. We start out with all values
		# equal to zero, meaning none of the pixel colors have been selected yet.
		self.pixel_values = np.full(self.dimension*self.dimension, 1)

	# The reward is a function of how much we were able to trick the discriminator (i.e. how
	# high the fake_prob is) and how many pixels had to be filled in.
	def _compute_reward(self, fake_prob, num_unfilled_pixels):
		# For now, we try the following reward:
		# - We want the fake_prob to be correlated with the reward
		# - We want the num_unfilled_pixels to be inversely weighted with the reward
		# So we just try fake_prob / (num_unfilled_pixels + 1).
		# Note the +1 is needed so we don't divide by zero.
		# TODO: Experiment with this.
		# Selecting this 0.48 value seemed to make learning a lot faster. Also setting the reward factor to be a lot bigger.

		a = fake_prob - 0.48
		if a > 0:
			a *= 10 # Strengthen correct signal reward.
		return a * REWARD_FACTOR / (num_unfilled_pixels + 1)

	# Returns a copied state with the remaining pixels filled in according to the current policy,
	# and also returns the number of pixels that had to be filled in.
	def _fill_remaining_pixels(self, pixels, fill_policy=None, debug=False):
		# TODO, actually do this properly.
		num_unfilled_pixels = 0
		copied_pixels = np.copy(pixels)
		# print("Filling in with fill policy")
		while not np.all(copied_pixels):
			num_unfilled_pixels += 1
			action_idx = fill_policy(copied_pixels)
			action = self._convert_action_idx_to_action(action_idx)
			if (copied_pixels[action[1]] != 0):
				# Returning early because we tried to fill an already filled pixel
				return copied_pixels, num_unfilled_pixels, True
			copied_pixels[action[1]] = action[0]
		if debug:
			print("Filled in with policy... rendering")
			self._print_pixels(copied_pixels)
		return copied_pixels, num_unfilled_pixels, False


		# for i in xrange(copied_pixels.size):
		# 	# For now, if we find an empty pixel, just fill it in with a random pixel.
		# 	if copied_pixels[i] == 0:
		# 		copied_pixels[i] = np.random.randint(1,3)
		# 		num_unfilled_pixels += 1
		# return copied_pixels, num_unfilled_pixels


