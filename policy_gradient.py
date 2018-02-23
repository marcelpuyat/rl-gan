import gym
import drawing_env
from drawing_env.envs.ops import *
import numpy as np
import tensorflow as tf

BETA1 = 0.5
LEARNING_RATE = 7e-4
BATCH_SIZE = 1
DIMENSION = 5
NUM_EPISODES = 100000
NUM_POSSIBLE_PIXEL_COLORS = 2
EPSILON_GREEDY_START = 0.25
EPSILON_GREEDY_PER_EPISODE_DECAY = 0.999
NUM_ACTIONS = DIMENSION*DIMENSION*NUM_POSSIBLE_PIXEL_COLORS
NUM_STATES = DIMENSION*DIMENSION
NUM_BATCHES = 100000
NUM_EPISODES_PER_BATCH = 1

env = gym.make('DrawEnv-v0')

# Create policy gradient TF graph.
pixels_placeholder = tf.placeholder(tf.float32, shape=[None, NUM_STATES])
actions_taken = tf.placeholder(tf.int32, shape=[None])
rewards = tf.placeholder(tf.float32, shape=[None])

batch_size = tf.shape(pixels_placeholder)[0]
reshaped_input = tf.reshape(pixels_placeholder, tf.stack([batch_size, DIMENSION, DIMENSION, 1]))

get_pixel_validation_layer = lrelu(conv2d(reshaped_input, 1, 1, 1, 1, 1, name="validation_layer"))
print(get_pixel_validation_layer.get_shape())
h0 = lrelu(conv2d(get_pixel_validation_layer, 4, 2, 2, 1, 1, name="conv1"))
h1 = lrelu(conv2d(h0, 8, 2, 2, 1, 1, name="conv2"))
h2 = lrelu(conv2d(h1, 16, 2, 2, 1, 1, name="conv3"))
h2_flatted = tf.reshape(h2, [batch_size, DIMENSION * DIMENSION * 16])
h3 = lrelu(dense(h2_flatted, NUM_ACTIONS * 2, name="dense1"))
h4 = dense(h3, NUM_ACTIONS, name="dense2")

# Convert log probs into actual probabilities
action_probs = tf.squeeze(tf.nn.softmax(h4))

# We want a one-hot vector for each of the actions we took. So we can scale the rewards by this
# and have a tensor to use for the loss function.
action_mask = tf.one_hot(actions_taken, NUM_ACTIONS, 1.0, 0.0)
action_predictions = tf.reduce_sum(action_mask * (action_probs + 1e-5), axis=1)

# Now we want to increase/decrease the chances of these actions occurring based on the reward for it.
# Note that there's a negative sign because we want to maximize this.
loss = -tf.reduce_mean(tf.log(action_predictions) * rewards)

optimizer = tf.train.AdamOptimizer(LEARNING_RATE, BETA1)
train_op = optimizer.minimize(loss)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	# Batch loop
	for batch in xrange(NUM_BATCHES):
		print("----------------")
		print("Train iter: " + str(batch))
		print("----------------")
		# Episode loop
		for episode in xrange(NUM_EPISODES_PER_BATCH):
			episode_actions = np.zeros(NUM_STATES)
			episode_rewards = np.zeros(NUM_STATES)
			episode_states = np.zeros((NUM_STATES, NUM_STATES))
			episode_reward = 0
			done = False
			iteration = 0
			last_action_was_valid = False
			state = env.reset()

			# Take action loop
			while done != True:
				action_dist = sess.run(action_probs, {pixels_placeholder: np.array([state])})
				print(action_dist)
				action_to_take = np.random.choice(NUM_ACTIONS, p=action_dist)
				print("Policy said to take action: " + str(action_to_take))
				if np.random.rand() < 0.10:
					print("Taking random action!")
					random_choices = np.argwhere(state == -1).flatten()
					action_to_take = random_choices[np.random.randint(0, random_choices.size)] * 2
					action_to_take += np.random.randint(0,NUM_POSSIBLE_PIXEL_COLORS)

				next_state, curr_reward, done, _, last_action_was_valid = env.step_with_random(action_to_take)

				episode_actions[iteration] = action_to_take
				episode_states[iteration] = state
				episode_rewards[iteration] = curr_reward

				episode_reward += curr_reward
				state = next_state
				iteration += 1

			# Look back at all the rewards. For all those that are zero (i.e. we made a pixel selection
			# that was valid), we want to get some signal for how good this was. Remember that the loop
			# above simply sets its reward to what the environment returns, which will always be zero
			# except for the last action that completes the image. So for all zero rewards, we want to
			# assign the reward that we saw in the last action that completed the image. Note that
			# we make sure this last action was in fact valid as well.
			if last_action_was_valid:
				for r_idx in xrange(episode_rewards.size):
					if episode_rewards[r_idx] == 0:
						episode_rewards[r_idx] = episode_rewards[-1]

			env.render()

			# # Set reward for all actions in this episode.
			# episode_rewards[:] = episode_reward

			print("Rewards: " + str(episode_rewards))
			print("States: " + str(episode_states))
			print("Actions: " + str(episode_actions))
			print("Training")
			# Train on batch data.
			_, loss_val = sess.run([train_op, loss], {actions_taken: episode_actions, pixels_placeholder: episode_states, rewards: episode_rewards})
			print("Loss: " + str(loss_val))



