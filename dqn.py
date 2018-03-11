import gym
import drawing_env
from drawing_env.envs.ops import *
import numpy as np
import tensorflow as tf
import math

BETA1 = 0.5
LEARNING_RATE = 5e-3
BATCH_SIZE = 1
LOCAL_DIMENSION = 3
FULL_DIMENSION = 6

NUM_EPISODES = 100000
NUM_STATES = LOCAL_DIMENSION*LOCAL_DIMENSION
EPISODE_LENGTH = FULL_DIMENSION*FULL_DIMENSION*2

NUM_POSSIBLE_PIXEL_COLORS = 2
EPSILON_GREEDY_START = 0.25
EPSILON_GREEDY_PER_EPISODE_DECAY = 0.9999

env = gym.make('DrawEnv-v0')

# This returns a LOCAL_DIMxLOCAL_DIM area around the given coordinate, mirroring pixels when we are on the edge.
# Haven't tried simply using zero padding.
def get_local_pixels(all_pixels, coord):
	reshaped = all_pixels.reshape((FULL_DIMENSION, FULL_DIMENSION))
	x_coord = coord % FULL_DIMENSION
	y_coord = int(coord) / FULL_DIMENSION
	local_pix = np.zeros((LOCAL_DIMENSION, LOCAL_DIMENSION))

	prev_row = y_coord-(LOCAL_DIMENSION-2)
	prev_col = x_coord-(LOCAL_DIMENSION-2)
	next_row = y_coord+(LOCAL_DIMENSION-2)
	next_col = x_coord+(LOCAL_DIMENSION-2)

	local_pix = np.zeros((LOCAL_DIMENSION,LOCAL_DIMENSION))
	padded_pixels = np.zeros((FULL_DIMENSION+(LOCAL_DIMENSION-2)*2, FULL_DIMENSION+(LOCAL_DIMENSION-2)*2))
	padded_pixels[1:FULL_DIMENSION+1,1:FULL_DIMENSION+1] = reshaped[:,:]

	# Check if we have to mirror the edges
	left_padding = x_coord
	if left_padding == 0:
		padded_pixels[prev_row+1:next_row+2,0] = padded_pixels[prev_row+1:next_row+2,2]
	right_padding = FULL_DIMENSION - 1 - (x_coord)
	if right_padding == 0:
		padded_pixels[prev_row+1:next_row+2,FULL_DIMENSION+(LOCAL_DIMENSION-2)*2 - 1] = padded_pixels[prev_row+1:next_row+2,FULL_DIMENSION+(LOCAL_DIMENSION-2)*2 - 3]
	top_padding = y_coord
	if top_padding == 0:
		padded_pixels[0,prev_col+1:next_col+2] = padded_pixels[2,prev_col+1:next_col+2]
	bottom_padding = FULL_DIMENSION - 1 - (y_coord)
	if bottom_padding == 0:
		padded_pixels[FULL_DIMENSION+(LOCAL_DIMENSION-2)*2 - 1,prev_col+1:next_col+2] = padded_pixels[FULL_DIMENSION+(LOCAL_DIMENSION-2)*2 - 3,prev_col+1:next_col+2]


	# Check for corner cases
	if left_padding == 0 and bottom_padding == 0:
		padded_pixels[FULL_DIMENSION+(LOCAL_DIMENSION-2)*2-1][0] = reshaped[FULL_DIMENSION-2][1]
	elif right_padding == 0 and bottom_padding == 0:
		padded_pixels[FULL_DIMENSION+(LOCAL_DIMENSION-2)*2-1][FULL_DIMENSION+(LOCAL_DIMENSION-2)*2-1] = reshaped[FULL_DIMENSION-2][FULL_DIMENSION-2]
	elif right_padding == 0 and top_padding == 0:
		padded_pixels[0][FULL_DIMENSION+(LOCAL_DIMENSION-2)*2-1] = reshaped[1][FULL_DIMENSION-2]
	elif left_padding == 0 and top_padding == 0:
		padded_pixels[0][0] = reshaped[1][1]

	local_pix[:,:] = padded_pixels[prev_row+1:next_row+2,prev_col+1:next_col+2]

	return local_pix.flatten()

# DQN Architecture.
def deep_q_network(pixels, coordinate, number, num_actions_per_pixel):
	with tf.variable_scope("deep_q_network") as scope:

		batch_size = tf.shape(pixels)[0]
		reshaped_input = tf.reshape(pixels, tf.stack([batch_size, LOCAL_DIMENSION, LOCAL_DIMENSION, 1]))

		h0 = lrelu(conv2d(reshaped_input, 4, 2, 2, 1, 1, name="conv1"))
		h1 = lrelu(conv2d(h0, 8, 2, 2, 1, 1, name="conv2"))
		h2 = lrelu(conv2d(h1, 16, 2, 2, 1, 1, name="conv3"))
		h2_flatted = tf.reshape(h2, [batch_size, LOCAL_DIMENSION * LOCAL_DIMENSION * 16])

		# Append coordinate and number label to last layer because we don't want it to be convoluted with
		# the pixel values.
		h3 = dense(tf.concat([h2_flatted, coordinate, number], axis=1), FULL_DIMENSION*FULL_DIMENSION, name="dense1")
		h4 = dense(h3, FULL_DIMENSION, name="dense2")
		h5 = dense(h4, num_actions_per_pixel, name="dense3")
		return h5

# Set up placeholders.
pixels_placeholder = tf.placeholder(tf.float32, shape=[None, LOCAL_DIMENSION*LOCAL_DIMENSION], name='state')
coordinate_placeholder = tf.placeholder(tf.float32, shape=[None, 1])
number_placeholder = tf.placeholder(tf.float32, shape=[None, 1])
action_rewards_placeholder = tf.placeholder(tf.float32, shape=[None], name='actual_reward')
action_selected_placeholder = tf.placeholder(tf.int32, shape=[None])

action_mask = tf.one_hot(action_selected_placeholder, 2, 1.0, 0.0)

# Set up loss function
estimated_q_value = deep_q_network(pixels_placeholder, coordinate_placeholder, number_placeholder, NUM_POSSIBLE_PIXEL_COLORS)
objective_fn = tf.reduce_mean(tf.square(action_rewards_placeholder - tf.reduce_sum(estimated_q_value * action_mask, axis=1)))
tf.summary.scalar("DQN Loss", objective_fn)
# Set up optimizer
q_optimizer = tf.train.AdamOptimizer(LEARNING_RATE, BETA1)
grads = q_optimizer.compute_gradients(objective_fn)
for grad, var in grads:
	if grad is not None:
		tf.summary.scalar(var.op.name + "/gradient", tf.reduce_mean(grad))
train_q_network = q_optimizer.apply_gradients(grads)

sess = tf.Session()
env.set_session(sess)

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('tensorboard/',
									 sess.graph)

# Initialize TF graph.
sess.run(tf.global_variables_initializer())

# Initialize epsilon greedy.
epsilon_greedy = EPSILON_GREEDY_START

# First train discriminator 100 iterations.
real_prob = 0.5
fake_prob = 0.5
num_iters = 0
while (real_prob < 0.75 and fake_prob > 0.25) or num_iters < 5000:
	fake_prob, real_prob = env.train_disc_random_fake()
	num_iters += 1

action_count = {}

# Training.
for i in xrange(NUM_EPISODES):
	tf.summary.scalar("epsilon greedy", epsilon_greedy)
	print("Episode num: " + str(i))
	curr_state = env.reset()
	episode_done = False

	pixels_batch = np.zeros((EPISODE_LENGTH,LOCAL_DIMENSION*LOCAL_DIMENSION))
	coordinates_batch = np.zeros((EPISODE_LENGTH, 1))
	numbers_batch = np.zeros((EPISODE_LENGTH, 1))
	actions_selected_batch = np.zeros(EPISODE_LENGTH)
	rewards = np.zeros(EPISODE_LENGTH)

	itr = 0
	# Perform one episode to finish.
	while not episode_done:
		all_pixels = curr_state['pixels']
		number = curr_state['number']

		local_pixels = get_local_pixels(all_pixels, curr_state['coordinate'])

		pixels_batch[itr] = local_pixels
		pixels_batch[itr+1] = local_pixels
		coordinates_batch[itr][0] = curr_state['coordinate']
		coordinates_batch[itr+1][0] = curr_state['coordinate']
		numbers_batch[itr][0] = number
		numbers_batch[itr+1][0] = number

		state_bytes = local_pixels.tobytes()

		# First do a forward prop to select the best action given the current state.
		q_value_estimates = sess.run([estimated_q_value], {number_placeholder: np.array([[number]]), pixels_placeholder: np.array([local_pixels]), coordinate_placeholder: np.array([[curr_state['coordinate']]]) })
		selected_action = np.argmax(q_value_estimates[0])
		other_action = 1 if selected_action == 0 else 0

		rand_prob = 0.0

		# Compute certainty of our action choice by seeing how many times we've taken it compared to the other action in this
		# particular state.
		if state_bytes not in action_count:
			action_count[state_bytes] = {}
			action_count[state_bytes][0] = 0
			action_count[state_bytes][1] = 0
		elif action_count[state_bytes][selected_action] != 0:
			print("Num times we've taken preferred in this state: " + str(action_count[state_bytes][selected_action]))
			print("Num times we've taken other in this state: " + str(action_count[state_bytes][other_action]))
			rand_prob = (0.30 - (float(action_count[state_bytes][other_action]) / \
				(action_count[state_bytes][other_action] + action_count[state_bytes][selected_action])))

			# Annealing based on episode. The layer the episode, the lower our random prob chance. Mostly just fiddled with these numbers.
			episode_annealing = (4 * (1 - (float(i) / NUM_EPISODES)))
			if episode_annealing == 0:
				episode_annealing = 1

			# Annealing based on state. We want some sort of log/sqrt type function that increases random prob chance as we move on to later coordinates (since we usually are easily able to learn to do well for the earlier ones).
			# Mostly just fiddled with these constants.
			state_annealing = math.sqrt((FULL_DIMENSION*FULL_DIMENSION - curr_state['coordinate'])*5)
			if state_annealing == 0:
				state_annealing = 1

			rand_prob /= state_annealing / episode_annealing

			print("Probability of switching: " + str(rand_prob))
			rand_prob = max(rand_prob, 0)

		if np.random.rand() < rand_prob:
			print("Selecting random action, rand prob: " + str(rand_prob))
			tmp = selected_action
			selected_action = other_action
			other_action = tmp

		actions_selected_batch[itr] = selected_action
		actions_selected_batch[itr+1] = other_action
		_, other_reward, _, _ = env.try_step(other_action)
		next_state, reward, episode_done, _ = env.step(selected_action)
		rewards[itr] = reward
		rewards[itr+1] = other_reward

		# Update action certainty
		action_count[state_bytes][selected_action] += 1

		curr_state = next_state
		itr += 2

	# shuffle all lists in batch
	random_order = np.random.choice(rewards.size, size=rewards.size)
	rewards = rewards[random_order]
	pixels_batch = pixels_batch[random_order]
	coordinates_batch = coordinates_batch[random_order]
	actions_selected_batch = actions_selected_batch[random_order]
	# Given the reward, train our DQN.
	discrim_real_placeholder, discrim_real_label_placeholder, discrim_fake_placeholder, discrim_fake_label_placeholder = env.get_discrim_placeholders()
	real_values, real_labels, fake_values, fake_labels = env.get_discrim_placeholder_values()
	real_loss, fake_loss = env.discrim_loss_tensors()
	d_r_loss, d_f_loss, summary, _, loss = sess.run([real_loss, fake_loss, merged, train_q_network, objective_fn], {discrim_real_label_placeholder: real_labels, discrim_fake_label_placeholder: fake_labels, number_placeholder: numbers_batch, pixels_placeholder: pixels_batch, action_rewards_placeholder: rewards, action_selected_placeholder: actions_selected_batch, coordinate_placeholder: coordinates_batch, discrim_real_placeholder: real_values, discrim_fake_placeholder: fake_values})
	print("Real loss: " + str(d_r_loss))
	print("Fake loss: " + str(d_f_loss))
	print("DQN Loss: " + str(loss))
	if i % 10 == 0:
		train_writer.add_summary(summary, i)
	print("Episode finished. Rendering:")
	env.render()

	i += 1
	if i % 100 == 0:
		state = env.reset()
		done = False
		# Do a full episode with no randomness
		while not done:
			local_pix = get_local_pixels(state['pixels'], state['coordinate'])
			q_value_estimates = sess.run([estimated_q_value], {number_placeholder: np.array([[state['number']]]), pixels_placeholder: np.array([local_pix]), coordinate_placeholder: np.array([[state['coordinate']]]) })
			selected_action = np.argmax(q_value_estimates[0])
			next_state, _, done, _ = env.step(selected_action)
			state = next_state
		print("--------------------")
		print("--------------------")
		print("Test with no randomness")
		env.render()
		print("--------------------")
		print("--------------------")

	# Decay epsilon greedy value
	epsilon_greedy *= EPSILON_GREEDY_PER_EPISODE_DECAY

