import gym
import drawing_env
from drawing_env.envs.ops import *
import numpy as np
import tensorflow as tf
import math

BETA1 = 0.5
LEARNING_RATE = 5e-3
BATCH_SIZE = 1
DIMENSION = 5
NUM_EPISODES = 100000
NUM_POSSIBLE_PIXEL_COLORS = 2
EPSILON_GREEDY_START = 0.25
EPSILON_GREEDY_PER_EPISODE_DECAY = 0.9999

env = gym.make('DrawEnv-v0')

# DQN Architecture
def deep_q_network(pixels, coordinate, num_actions_per_pixel):
	with tf.variable_scope("deep_q_network") as scope:

		batch_size = tf.shape(pixels)[0]
		reshaped_input = tf.reshape(pixels, tf.stack([batch_size, DIMENSION, DIMENSION, 1]))

        # Outputs 2 * num pixels, so mod by two to get whether it's black or white, and divide by 2 to get the num action.
		h0 = lrelu(conv2d(reshaped_input, 4, 2, 2, 1, 1, name="conv1"))
		h1 = lrelu(conv2d(h0, 8, 2, 2, 1, 1, name="conv2"))
		h2 = lrelu(conv2d(h1, 16, 2, 2, 1, 1, name="conv3"))
		h2_flatted = tf.reshape(h2, [batch_size, DIMENSION * DIMENSION * 16])
		h3 = dense(tf.concat([h2_flatted, coordinate], axis=1), num_actions_per_pixel, name="dense2")
		return h3

# Set up placeholder
pixels_placeholder = tf.placeholder(tf.float32, shape=[None, DIMENSION*DIMENSION], name='state')
coordinate_placeholder = tf.placeholder(tf.float32, shape=[None, 1])
action_rewards_placeholder = tf.placeholder(tf.float32, shape=[None], name='actual_reward')
action_selected_placeholder = tf.placeholder(tf.int32, shape=[None])

action_mask = tf.one_hot(action_selected_placeholder, 2, 1.0, 0.0)

# Set up loss function
estimated_q_value = deep_q_network(pixels_placeholder, coordinate_placeholder, NUM_POSSIBLE_PIXEL_COLORS)
objective_fn = tf.reduce_sum(tf.square(action_rewards_placeholder - tf.reduce_sum(estimated_q_value * action_mask, axis=1)))
tf.summary.scalar("DQN Loss", objective_fn)
# Set up optimizer
q_optimizer = tf.train.AdamOptimizer(LEARNING_RATE, BETA1)
grads = q_optimizer.compute_gradients(objective_fn)
for grad, var in grads:
	if grad is not None:
		tf.summary.scalar(var.op.name + "/gradient", tf.reduce_mean(grad))
train_q_network = q_optimizer.apply_gradients(grads)

sess = tf.Session()

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('td_tensorboard/',
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

	pixels_batch = np.zeros((DIMENSION*DIMENSION*2,DIMENSION*DIMENSION))
	coordinates_batch = np.zeros((DIMENSION*DIMENSION*2, 1))
	actions_selected_batch = np.zeros(DIMENSION*DIMENSION*2)
	rewards = np.zeros(DIMENSION*DIMENSION*2)

	itr = 0
	# Perform one episode to finish.
	while not episode_done:
		pixels_batch[itr] = curr_state['pixels']
		pixels_batch[itr+1] = curr_state['pixels']
		coordinates_batch[itr][0] = curr_state['coordinate']
		coordinates_batch[itr+1][0] = curr_state['coordinate']

		state_bytes = curr_state['pixels'].tobytes()

		# First do a forward prop to select the best action given the current state.
		q_value_estimates = sess.run([estimated_q_value], {pixels_placeholder: np.array([curr_state['pixels']]), coordinate_placeholder: np.array([[curr_state['coordinate']]]) })
		selected_action = np.argmax(q_value_estimates[0])
		other_action = 1 if selected_action == 0 else 0

		rand_prob = 0.0
		# Compute certainty
		if state_bytes not in action_count:
			action_count[state_bytes] = {}
			action_count[state_bytes][0] = 0
			action_count[state_bytes][1] = 0
		elif action_count[state_bytes][selected_action] != 0:
			print("Num times we've taken preferred in this state: " + str(action_count[state_bytes][selected_action]))
			print("Num times we've taken other in this state: " + str(action_count[state_bytes][other_action]))
			rand_prob = (1 - action_count[state_bytes][other_action] / action_count[state_bytes][selected_action])
			rand_prob /= math.sqrt(action_count[state_bytes][selected_action])
			print("Probability of switching: " + str(rand_prob))
			rand_prob = min(rand_prob, 0.30)

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
		itr += 1

	# shuffle all lists in batch
	random_order = np.random.choice(rewards.size, size=rewards.size)
	rewards = rewards[random_order]
	pixels_batch = pixels_batch[random_order]
	coordinates_batch = coordinates_batch[random_order]
	actions_selected_batch = actions_selected_batch[random_order]
	# Given the reward, train our DQN.
	_, loss = sess.run([train_q_network, objective_fn], {pixels_placeholder: pixels_batch, action_rewards_placeholder: rewards, action_selected_placeholder: actions_selected_batch, coordinate_placeholder: coordinates_batch})
	print("DQN Loss: " + str(loss))
	if loss > 50:
		print("Retraining on batch because of high loss")
		_, loss = sess.run([train_q_network, objective_fn], {pixels_placeholder: pixels_batch, action_rewards_placeholder: rewards, action_selected_placeholder: actions_selected_batch, coordinate_placeholder: coordinates_batch})
		print("Retrained loss: " + str(loss))
	# train_writer.add_summary(summary, i)
	print("")
	print("Episode finished. Rendering:")
	env.render()

	i += 1
	if i % 100 == 0:
		state = env.reset()
		done = False
		# Do a full episode with no randomness
		while not done:
			q_value_estimates = sess.run([estimated_q_value], {pixels_placeholder: np.array([state['pixels']]), coordinate_placeholder: np.array([[state['coordinate']]]) })
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

# # Try and draw an image with no randomness and no learning
# print("Drawing as a test")
# curr_state = env.reset()
# episode_done = False
# while not episode_done:
# 	state_batch = np.array([curr_state])

# 	print("Current state: " + str(state_batch))

# 	# First do a forward prop to select the best action given the current state.
# 	q_value_estimates = sess.run([estimated_q_value], {state_placeholder: state_batch})

# 	print("Estimated Q Values: " + str(q_value_estimates))

# 	reward_batch = np.zeros((BATCH_SIZE,1))
# 	action_selection_batch = np.zeros((BATCH_SIZE,1))
# 	# For now, let's just do a loop over all the items in a batch...
# 	for b in xrange(BATCH_SIZE):

# 		max_idx = np.argmax(q_value_estimates[b][0])
# 		best_action_num_value = q_value_estimates[b][0][max_idx]

# 		# Remember best_action_num_value is in the 1d flattened range of num_pixels * num_actions_per_pixel so
# 		# we have to do divison and modulus to fetch out the actual best pixel color and coordinate.
# 		best_pixel_color, best_pixel_coordinate = int((max_idx % 2) + 1), int(max_idx / 2)
# 		print("Best color: " + str(best_pixel_color))
# 		print("Best pixel coord: " + str(best_pixel_coordinate))

# 		# Using the best action, take it and get the reward and set the next state.
# 		curr_state, reward, episode_done, _ = env.step_with_fill_policy([best_pixel_color, best_pixel_coordinate])
# 		print("Reward seen for this action: " + str(reward))
# 		print("Reward we thought this action would have: " + str(q_value_estimates[b][0][int(max_idx)]))
# 		reward_batch[b] = reward
# 		action_selection_batch[b] = max_idx
# print("Test episode finished")
# env.render()


