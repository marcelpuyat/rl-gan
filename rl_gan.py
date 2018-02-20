import gym
import drawing_env
from drawing_env.envs.ops import *
import numpy as np
import tensorflow as tf

BETA1 = 0.5
LEARNING_RATE = 1e-3
BATCH_SIZE = 1
DIMENSION = 4
NUM_EPISODES = 50000
NUM_POSSIBLE_PIXEL_COLORS = 2
EPSILON_GREEDY_START = 0.3
EPSILON_GREEDY_PER_EPISODE_DECAY = 0.999

env = gym.make('DrawEnv-v0')

# DQN Architecture
def deep_q_network(state, num_pixels, num_actions_per_pixel):
	with tf.variable_scope("deep_q_network") as scope:


		reshaped_input = tf.reshape(state, [BATCH_SIZE, DIMENSION, DIMENSION, 1])

        # Outputs 2 * num pixels, so mod by two to get whether it's black or white, and divide by 2 to get the num action.
		h0 = lrelu(conv2d(reshaped_input, 4, 2, 2, 1, 1, name="conv1"))
		h1 = lrelu(conv2d(h0, 8, 2, 2, 1, 1, name="conv2"))
		h2 = lrelu(conv2d(h1, 16, 2, 2, 1, 1, name="conv3"))
		h2_flatted = tf.reshape(h2, [BATCH_SIZE, DIMENSION * DIMENSION * 16])
		h3 = dense(h2_flatted, num_pixels * num_actions_per_pixel, name="dense2")
		return h3

def select_best_action_idx(state, q_value_tensor, state_placeholder_tensor):
	state_batch = np.array([state])
	q_value_estimates = sess.run([q_value_tensor], {state_placeholder_tensor: state_batch})
	non_zero_states = np.argwhere(state != 0)
	non_zero_actions = np.append(non_zero_states * 2, non_zero_states * 2 + 1)
	q_value_estimates[b][0][non_zero_actions] = -float('inf')
	max_idx = np.argmax(q_value_estimates[b][0])
	return max_idx

def convert_action_idx_to_action(action_idx):
	# Remember action_idx is in the 1d flattened range of num_pixels * num_actions_per_pixel so
	# we have to do divison and modulus to fetch out the actual best pixel color and coordinate.
	return int((action_idx % 2) + 1), int(action_idx / 2)


# Set up placeholder
state_placeholder = tf.placeholder(tf.float32, shape=[BATCH_SIZE, DIMENSION*DIMENSION], name='state')
actual_reward = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 1], name='actual_reward')
action_selection = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 1], name='action_selection')

# Set up loss function
estimated_q_value = deep_q_network(state_placeholder, DIMENSION*DIMENSION, NUM_POSSIBLE_PIXEL_COLORS)
objective_fn = tf.losses.mean_squared_error(actual_reward, tf.gather(estimated_q_value, action_selection, axis=1)[0])
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
train_writer = tf.summary.FileWriter('tensorboard/',
									 sess.graph)

# Initialize TF graph.
sess.run(tf.global_variables_initializer())

# Initialize epsilon greedy.
epsilon_greedy = EPSILON_GREEDY_START

# Training.
for i in xrange(NUM_EPISODES):
	print("Episode num: " + str(i))
	curr_state = env.reset()
	episode_done = False

	# Perform one episode to finish.
	while not episode_done:
		# TODO: Use replay buffer to batch data.
		state_batch = np.array([curr_state])

		print("Current state: " + str(state_batch))

		# First do a forward prop to select the best action given the current state.
		q_value_estimates = sess.run([estimated_q_value], {state_placeholder: state_batch})

		print("Estimated Q Values: " + str(q_value_estimates))

		reward_batch = np.zeros((BATCH_SIZE,1))
		action_selection_batch = np.zeros((BATCH_SIZE,1))
		# For now, let's just do a loop over all the items in a batch...
		for b in xrange(BATCH_SIZE):
			s = state_batch[b]
			non_zero_states = np.argwhere(s != 0)
			print("Non zero states: " + str(non_zero_states))
			non_zero_actions = np.append(non_zero_states * 2, non_zero_states * 2 + 1)

			print("Q value estimates prior to validation: " + str (q_value_estimates[b][0]))
			q_value_estimates[b][0][non_zero_actions] = -float('inf')
			print("Q value estimates after validation: " + str (q_value_estimates[b][0]))
			max_idx = np.argmax(q_value_estimates[b][0])

			# TODO: Make this GLIE
			if np.random.rand() < epsilon_greedy:
				print("Selecting random action!")
				print(epsilon_greedy)
				random_choices = np.argwhere(s == 0)
				print("Selecting from zero-filled values")
				print(random_choices)
				rand_idx = np.random.randint(0, random_choices.size)
				# Smart random selection: only select pixel coords that are not yet selected.
				max_idx = random_choices[rand_idx][0] * 2
				max_idx += np.random.randint(0,2)
				print("Selecting: ")
				print(max_idx)

			best_pixel_color, best_pixel_coordinate = convert_action_idx_to_action(max_idx)
			print("Best color: " + str(best_pixel_color))
			print("Best pixel coord: " + str(best_pixel_coordinate))

			# Using the best action, take it and get the reward and set the next state.
			next_state, reward, episode_done, _ = env.step_with_fill_policy([best_pixel_color, best_pixel_coordinate], lambda s: convert_action_idx_to_action(select_best_action_idx(s, estimated_q_value, state_placeholder)))
			print("Reward seen for this action: " + str(reward))
			print("Reward we thought this action would have: " + str(q_value_estimates[b][0][int(max_idx)]))
			reward_batch[b] = reward

			# next_state_q_values = sess.run([estimated_q_value], {state_placeholder: np.array([next_state])})

			# reward_batch[b] += np.max(next_state_q_values[0][0])
			# print("Q(s',a') = " + str(np.max(next_state_q_values[0][0])))

			curr_state = next_state 
			action_selection_batch[b] = max_idx

		print("Training with the following:")
		print("\t\tState: " + str(state_batch))
		print("\t\tReward: " + str(reward_batch))
		print("\t\tReward we thought this action would have: " + str(q_value_estimates[b][0][int(max_idx)]))
		print("\t\tSelected action: " + str(action_selection_batch))
		# Given the reward, train our DQN.
		_, loss, summary = sess.run([train_q_network, objective_fn, merged], {state_placeholder: state_batch, actual_reward: reward_batch, action_selection: action_selection_batch})
		print("DQN Loss: " + str(loss))
		train_writer.add_summary(summary, i)
		print("")
	print("Episode finished. Rendering:")
	env.render()

	# Decay epsilon greedy value
	epsilon_greedy *= EPSILON_GREEDY_PER_EPISODE_DECAY

# Try and draw an image with no randomness and no learning
print("Drawing as a test")
curr_state = env.reset()
episode_done = False
while not episode_done:
	state_batch = np.array([curr_state])

	print("Current state: " + str(state_batch))

	# First do a forward prop to select the best action given the current state.
	q_value_estimates = sess.run([estimated_q_value], {state_placeholder: state_batch})

	print("Estimated Q Values: " + str(q_value_estimates))

	reward_batch = np.zeros((BATCH_SIZE,1))
	action_selection_batch = np.zeros((BATCH_SIZE,1))
	# For now, let's just do a loop over all the items in a batch...
	for b in xrange(BATCH_SIZE):

		max_idx = np.argmax(q_value_estimates[b][0])
		best_action_num_value = q_value_estimates[b][0][max_idx]

		# Remember best_action_num_value is in the 1d flattened range of num_pixels * num_actions_per_pixel so
		# we have to do divison and modulus to fetch out the actual best pixel color and coordinate.
		best_pixel_color, best_pixel_coordinate = int((max_idx % 2) + 1), int(max_idx / 2)
		print("Best color: " + str(best_pixel_color))
		print("Best pixel coord: " + str(best_pixel_coordinate))

		# Using the best action, take it and get the reward and set the next state.
		curr_state, reward, episode_done, _ = env.step_with_fill_policy([best_pixel_color, best_pixel_coordinate])
		print("Reward seen for this action: " + str(reward))
		print("Reward we thought this action would have: " + str(q_value_estimates[b][0][int(max_idx)]))
		reward_batch[b] = reward
		action_selection_batch[b] = max_idx
print("Test episode finished")
env.render()


