from config import *
from utils import * 
import gym
import drawing_env
from drawing_env.envs.ops import *
import numpy as np
import tensorflow as tf

env = gym.make('DrawEnv-v0')

# Policy gradient architecture. Return logits for each action's probability
def policy_network(pixels, coordinate, number, num_actions_per_pixel):
  batch_size = tf.shape(pixels)[0]
  reshaped_input = tf.reshape(pixels, tf.stack([batch_size, LOCAL_DIMENSION, LOCAL_DIMENSION, 1]))
    
  h0 = lrelu(conv2d(reshaped_input, 4, 2, 2, 1, 1, name="conv1"))
  h1 = lrelu(conv2d(h0, 8, 2, 2, 1, 1, name="conv2"))
  h2 = lrelu(conv2d(h1, 16, 2, 2, 1, 1, name="conv3"))
  h2_flatted = tf.flatten(h2)
    
  # Append coordinate and number label to last layer because we don't want it to be convoluted with
  # the pixel values.
  h3 = tf.contrib.layers.fully_connected(tf.concat([h2_flatted, coordinate, number], axis=1),
                                         FULL_DIMENSION*FULL_DIMENSION, name='dense1')
  h4 = tf.contrib.layers.fully_connected(h3, FULL_DIMENSION, name='dense2')
  output = tf.contrib.layers.fully_connected(layer, num_actions_per_pixel,
                                               activation_fn=None, name='dense3') 
  return output


class PG(object):
  """
  Abstract Class for implementing a Policy Gradient Based Algorithm
  """
  def __init__(self, env, output_path, model_path, log_path, lr=LEARNING_RATE,
               use_baseline=True, normalize_advantage=True, batch_size=BATCH_SIZE,
               summary_freq=PG_SUMMARY_FREQ):
    """
    Initialize Policy Gradient Class
  
    Args:
            env: the open-ai environment
            config: class with hyperparameters
            logger: logger instance from logging module

    You do not need to implement anything in this function. However,
    you will need to use self.discrete, self.observation_dim,
    self.action_dim, and self.lr in other methods.
    
    """
    # directory for training outputs
    if not os.path.exists(output_path):
      os.makedirs(output_path)
    if not os.path.exists(model_path):
      os.makedirs(model_path)
            
    # store hyper-params
    self.config = config
    self.logger = get_logger(config.log_path)
    self.env = env

    # action dim
    # discrete action space or continuous action space
    self.action_dim = self.env.action_space.n # TODO: ensure this works
    self.lr = lr
  
    # build model
    self.build()

    
  def add_placeholders_op(self):
    """
    Adds placeholders to the graph
    Set up the observation, action, and advantage placeholder
    """
    self.pixels_placeholder = tf.placeholder(tf.float32, shape=(None, LOCAL_DIMENSION*LOCAL_DIMENSION),
                                             name='pixel_window')
    self.coordinate_placeholder = tf.placeholder(tf.float32, shape=(None, 1), name='current_coordinate')
    self.number_placeholder = tf.placeholder(tf.float32, shape=(None, 1), name='digit')
    
    self.taken_action_placeholder = tf.placeholder(tf.int32, shape=(None,), name='taken_action')
    self.advantage_palceolder = tf.placeholder(tf.float32, shape=(None,), name='advantage')
  
  
  def build_policy_network_op(self, scope="policy_network"):
    """
    Build the policy network, construct the tensorflow operation to sample 
    actions from the policy network outputs, and compute the log probabilities
    of the taken actions (for computing the loss later). These operations are 
    stored in self.sampled_action and self.logprob. Must handle both settings
    of self.discrete.
    """
    action_logits = policy_network(pixels_placeholder, coordinate_placeholder, number_placeholder,
                                   self.action_dim)
    self.sampled_action = tf.squeeze(tf.multinomial(action_logits, 1), name='sampled_action_discrete')
    self.logprob = -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.taken_action_placeholder,
                    logits=action_logits, name='taken_action_logprob_discrete')            
  
  
  def add_loss_op(self):
    """
    Sets the loss of a batch, the loss is a scalar 
    """
    # REINFORCE update uses mean over all trajectories of sum over each trajectory of log π * A_t
    self.pg_loss = -tf.reduce_mean(tf.multiply(self.logprob, self.advantage_placeholder), name='loss')
  
  
  def add_optimizer_op(self):
    """
    Sets the optimizer using AdamOptimizer
    """
    self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.pg_loss)
  
  
  def add_baseline_op(self, scope = "baseline"):
    """
    Build the baseline network within the scope
    """
    # policy_network returns (batch_size, 1) but targets fed as (batch_size,), so squeeze
    self.baseline = tf.squeeze(policy_network(pixels_placeholder, coordinate_placeholder,
                                              number_placeholder, 1), axis=1)
    self.baseline_target_placeholder = tf.placeholder(tf.float32, shape=(None,), name='baseline_target')
    loss = tf.losses.mean_squared_error(self.baseline_target_placeholder, self.baseline, scope=scope)
    self.update_baseline_op = tf.train.AdamOptimizer().minimize(loss, name='calibrate_baseline')

  
  def build(self):
    """
    Build model by adding all necessary variables

    You don't have to change anything here - we are just calling
    all the operations you already defined to build the tensorflow graph.
    """
  
    # add placeholders
    self.add_placeholders_op()
    # create policy net
    self.build_policy_network_op()
    # add square loss
    self.add_loss_op()
    # add optmizer for the main networks
    self.add_optimizer_op()
  
    if self.config.use_baseline:
      self.add_baseline_op()
  
  def initialize(self):
    """
    Assumes the graph has been constructed (have called self.build())
    Creates a tf Session and run initializer of variables

    You don't have to change or use anything here.
    """
    # create tf session
    self.sess = tf.Session()
    # tensorboard stuff
    self.add_summary()
    # initiliaze all variables
    init = tf.global_variables_initializer()
    self.sess.run(init)
  
  
  def add_summary(self):
    """
    Tensorboard stuff. 
    """
    # extra placeholders to log stuff from python
    self.avg_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="avg_reward")
    self.max_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="max_reward")
    self.std_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="std_reward")
    self.eval_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="eval_reward")
  
    # extra summaries from python -> placeholders
    tf.summary.scalar("Avg Reward", self.avg_reward_placeholder)
    tf.summary.scalar("Max Reward", self.max_reward_placeholder)
    tf.summary.scalar("Std Reward", self.std_reward_placeholder)
    tf.summary.scalar("Eval Reward", self.eval_reward_placeholder)
            
    # logging
    self.merged = tf.summary.merge_all()
    self.file_writer = tf.summary.FileWriter(self.config.output_path,self.sess.graph) 

    
  def init_averages(self):
    """
    Defines extra attributes for tensorboard.

    You don't have to change or use anything here.
    """
    self.avg_reward = 0.
    self.max_reward = 0.
    self.std_reward = 0.
    self.eval_reward = 0.
  

  def update_averages(self, rewards, scores_eval):
    """
    Update the averages.
  
    Args:
            rewards: deque
            scores_eval: list
    """
    self.avg_reward = np.mean(rewards)
    self.max_reward = np.max(rewards)
    self.std_reward = np.sqrt(np.var(rewards) / len(rewards))
  
    if len(scores_eval) > 0:
      self.eval_reward = scores_eval[-1]
  
  
  def record_summary(self, t):
    """
    Add summary to tfboard
    """
  
    fd = {
      self.avg_reward_placeholder: self.avg_reward, 
      self.max_reward_placeholder: self.max_reward, 
      self.std_reward_placeholder: self.std_reward, 
      self.eval_reward_placeholder: self.eval_reward, 
    }
    summary = self.sess.run(self.merged, feed_dict=fd)
    self.file_writer.add_summary(summary, t)


  def init_discriminator(self):
    ''' 
    Initialize the DrawEnv's discriminator with some training. 
    '''
    real_prob = 0.5
    fake_prob = 0.5
    for _ in range(5000):
      fake_prob, real_prob = env.train_disc_random_fake()
      if real_prob >= 0.75 or fake_prob <= 0.25:
        break
      
  
  def sample_path(self, env, num_episodes = None):
    """
    Sample path for the environment.
  
    Args:
            num_episodes:   the number of episodes to be sampled 
              if none, sample one batch (size indicated by config file)
    Returns:
        paths: a list of paths. Each path in paths is a dictionary with
            path["pixels"] a numpy array of ordered observations in the path
            path["coords"] a numpy array of the current coordinate in the path
            path["numbers"] a numpy array of the digit being created
            path["actions"] a numpy array of the corresponding actions in the path
            path["reward"] a numpy array of the corresponding rewards in the path
        total_rewards: the sum of all rewards encountered during this "path"
    """
    episode = 0
    episode_rewards = []
    paths = []
    t = 0
  
    while (num_episodes or t < self.config.batch_size):
      state = env.reset()
      states, actions, rewards = [], [], []
      episode_reward = 0
  
      for step in range(self.config.max_ep_len):
        states.append(state)
        # TODO: Orig code had [0] appended to end of line which causes crash with discrete actions. Removing; see if needed for cont actions
        action = self.sess.run(self.sampled_action, feed_dict={self.observation_placeholder : states[-1][None]})
        state, reward, done, info = env.step(action)
        actions.append(action)
        rewards.append(reward)
        episode_reward += reward
        t += 1
        if (done or step == self.config.max_ep_len-1):
          episode_rewards.append(episode_reward)  
          break
        if (not num_episodes) and t == self.config.batch_size:
          break
  
      path = {"observation" : np.array(states), 
                      "reward" : np.array(rewards), 
                      "action" : np.array(actions)}
      paths.append(path)
      episode += 1
      if num_episodes and episode >= num_episodes:
        break        
  
    return paths, episode_rewards
  
  
  def get_returns(self, paths):
    """
    Calculate the returns G_t for each timestep
  
    Args:
      paths: recorded sampled path.  See sample_path() for details.
  
    After acting in the environment, we record the observations, actions, and
    rewards. To get the advantages that we need for the policy update, we have
    to convert the rewards into returns, G_t, which are themselves an estimate
    of Q^π (s_t, a_t):
    
       G_t = r_t + γ r_{t+1} + γ^2 r_{t+2} + ... + γ^{T-t} r_T
    
    where T is the last timestep of the episode.

    TODO: compute and return G_t for each timestep. Use config.gamma.
    """

    all_returns = []
    for path in paths:
      rewards = path["reward"]
      #######################################################
      #########   YOUR CODE HERE - 5-10 lines.   ############
      returns = []
      for r in rewards[::-1]: # G[t-1] = r + γG[t], where G[t+1] = 0
        returns.append(r + config.gamma * (0 if not returns else returns[-1])) 
      returns = returns[::-1]
      #######################################################
      #########          END YOUR CODE.          ############
      all_returns.append(returns)
    returns = np.concatenate(all_returns)
  
    return returns
  
  
  def calculate_advantage(self, returns, observations):
    """
    Calculate the advantage
    Args:
            returns: all discounted future returns for each step
            observations: observations
              Calculate the advantages, using baseline adjustment if necessary,
              and normalizing the advantages if necessary.
              If neither of these options are True, just return returns.

    TODO:
    If config.use_baseline = False and config.normalize_advantage = False,
    then the "advantage" is just going to be the returns (and not actually
    an advantage). 

    if config.use_baseline, then we need to evaluate the baseline and subtract
      it from the returns to get the advantage. 
      HINT: 1. evaluate the self.baseline with self.sess.run(...

    if config.normalize_advantage:
      after doing the above, normalize the advantages so that they have a mean of 0
      and standard deviation of 1.
  
    """
    adv = returns
    #######################################################
    #########   YOUR CODE HERE - 5-10 lines.   ############
    if self.config.use_baseline:
      bl = self.sess.run(self.baseline, {self.observation_placeholder: observations})
      adv = returns - bl 
    if self.config.normalize_advantage:
      adv = (adv - np.mean(adv))/np.std(adv)
    #######################################################
    #########          END YOUR CODE.          ############
    return adv
  
  
  def update_baseline(self, returns, observations):
    """
    Update the baseline

    TODO:
      apply the baseline update op with the observations and the returns.
    """
    #######################################################
    #########   YOUR CODE HERE - 1-5 lines.   #############
    self.sess.run(self.update_baseline_op, {self.observation_placeholder: observations,
                                            self.baseline_target_placeholder: returns})
    #######################################################
    #########          END YOUR CODE.          ############
  
  
  def train(self):
    """
    Performs training

    You do not have to change or use anything here, but take a look
    to see how all the code you've written fits together!
    """
    last_eval = 0 
    last_record = 0
    scores_eval = []
    
    self.init_averages()
    scores_eval = [] # list of scores computed at iteration time
  
    for t in range(self.config.num_batches):
      # collect a minibatch of samples
      paths, total_rewards = self.sample_path(self.env) 
      scores_eval = scores_eval + total_rewards
      observations = np.concatenate([path["observation"] for path in paths])
      actions = np.concatenate([path["action"] for path in paths])
      # InvertedPendulum and HalfCheetah have spurious second dimension, so remove
      if not self.discrete:
        actions = actions[:,0,:]
      rewards = np.concatenate([path["reward"] for path in paths])
      # compute Q-val estimates (discounted future returns) for each time step
      returns = self.get_returns(paths)      
      advantages = self.calculate_advantage(returns, observations)

      # run training operations
      if self.config.use_baseline:
        self.update_baseline(returns, observations)
      self.sess.run(self.train_op, feed_dict={
                    self.observation_placeholder : observations, 
                    self.action_placeholder : actions, 
                    self.advantage_placeholder : advantages})
  
      # tf stuff: record summary and update saved model weights
      if (t % self.config.summary_freq == 0):
        self.update_averages(total_rewards, scores_eval)
        self.record_summary(t)
        # save model params
        self.saver = tf.train.Saver()
        self.saver.save(self.sess, config.model_path)

      # compute reward statistics for this batch and log
      avg_reward = np.mean(total_rewards)
      sigma_reward = np.sqrt(np.var(total_rewards) / len(total_rewards))
      msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
      self.logger.info(msg)
  
      if  self.config.record and (last_record > self.config.record_freq):
        self.logger.info("Recording...")
        last_record =0
        self.record()
  
    self.logger.info("- Training done.")
    export_plot(scores_eval, "Score", config.env_name, self.config.plot_output)


  def evaluate(self, env=None, num_episodes=1):
    """
    Evaluates the return for num_episodes episodes.
    Not used right now, all evaluation statistics are computed during training 
    episodes.
    """
    if env==None: env = self.env
    paths, rewards = self.sample_path(env, num_episodes)
    avg_reward = np.mean(rewards)
    sigma_reward = np.sqrt(np.var(rewards) / len(rewards))
    msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
    self.logger.info(msg)
    return avg_reward
     
  
  def record(self):
     """
     Re create an env and record a video for one episode
     """
     env = gym.make(self.config.env_name)
     env = gym.wrappers.Monitor(env, self.config.record_path, video_callable=lambda x: True, resume=True)
     self.evaluate(env, 1)
  

  def run(self):
    """
    Apply procedures of training for a PG.
    """
    # initialize
    self.initialize()

    # model
    self.train()


        

        
# TODO: Setup one for each agent?
with tf.variable_scope('pg_agent'):
        # Set up placeholders.
        pixels_placeholder = tf.placeholder(tf.float32, shape=[None, LOCAL_DIMENSION*LOCAL_DIMENSION],
                                            name='state')
        coordinate_placeholder = tf.placeholder(tf.float32, shape=[None, 1])
        number_placeholder = tf.placeholder(tf.float32, shape=[None, 1])

        taken_action_placeholder = tf.placeholder(tf.int32, shape=(None,), name='taken_action')
        advantage_placeholder = tf.placeholder(tf.float32, shape=(None,), name='advantage')

        # Get probabilities of actions from a given state using NN for action logits.
        action_logits = policy_network(pixels_placeholder, coordinate_placeholder, number_placeholder,
                                       NUM_POSSIBLE_PIXEL_COLORS)
        sampled_action = tf.squeeze(tf.multinomial(action_logits, 1), name='sampled_action_discrete')

        # Get log probability of action for policy gradient loss, then set loss and optiizer
        logprob = -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=taken_action_placeholder,
                   logits=action_logits, name='taken_action_logprob_discrete')
        pg_loss = -tf.reduce_mean(tf.multiply(logprob, advantage_placeholder), name='loss')
        train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(pg_loss)

        # Setup baseline computations.
        # Use same architecture as action logits to build baseline for state baseline
        # Returned tensor has shape (batch_size, 1), so squeeze last dim
        # TODO: Try a different architecture?
        baseline = tf.squeeze(policy_network(pixels_placeholder, coordinate_placeholder,
                                             number_placeholder, 1), axis=1)
        baseline_target_placeholder = tf.placeholder(tf.float32, shape=(None,), name='baseline_target')
        loss = tf.losses.mean_squared_error(baseline_target_placeholder, baseline)
        update_baseline_op = tf.train.AdamOptimizer().minimize(loss, name='calibrate_baseline')


########### TENSORFLOW SETUP ##########
tb_output = 'tensorboard/pg/'
model_path = tb_output + 'weights/'

# Initialize session and assign to environment.
sess = tf.Session()
env.set_session(sess)

# Add summary scalars. TODO: Setup one for each agent?
avg_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="pg_avg_reward")
max_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="pg_max_reward")
std_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="pg_std_reward")
eval_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="pg_eval_reward")

# extra summaries from python -> placeholders
tf.summary.scalar("PG Avg Reward", avg_reward_placeholder)
tf.summary.scalar("PG Max Reward", max_reward_placeholder)
tf.summary.scalar("PG Std Reward", std_reward_placeholder)
tf.summary.scalar("PG Eval Reward", eval_reward_placeholder)

# Logging
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(tb_output, sess.graph)

# Initialize TF graph.
sess.run(tf.global_variables_initializer())

self.avg_reward = 0.
self.max_reward = 0.
self.std_reward = 0.
self.eval_reward = 0.
  

def update_averages(self, rewards, scores_eval):
    """
    Update the averages.

    You don't have to change or use anything here.
  
    Args:
            rewards: deque
            scores_eval: list
    """
    self.avg_reward = np.mean(rewards)
    self.max_reward = np.max(rewards)
    self.std_reward = np.sqrt(np.var(rewards) / len(rewards))
  
    if len(scores_eval) > 0:
      self.eval_reward = scores_eval[-1]
  
  
  def record_summary(self, t):
    """
    Add summary to tfboard

    You don't have to change or use anything here.
    """
  
    fd = {
      self.avg_reward_placeholder: self.avg_reward, 
      self.max_reward_placeholder: self.max_reward, 
      self.std_reward_placeholder: self.std_reward, 
      self.eval_reward_placeholder: self.eval_reward, 
    }
    summary = self.sess.run(self.merged, feed_dict=fd)
    # tensorboard stuff
    self.file_writer.add_summary(summary, t)






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
		q_value_estimates = sess.run([estimated_q_value],
                                             {number_placeholder: np.array([[number]]),
                                              pixels_placeholder: np.array([local_pixels]),
                                              coordinate_placeholder: np.array([[curr_state['coordinate']]])
                                              })
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

			# Annealing based on state. We want some sort of log/sqrt type function that increases random prob chance
                        # as we move on to later coordinates (since we usually are easily able to learn to do well for the earlier ones).
			# Mostly just fiddled with these constants.
			state_annealing = np.sqrt((FULL_DIMENSION*FULL_DIMENSION - curr_state['coordinate'])*5)
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
	numbers_batch = numbers_batch[random_order]
        
	# Given the reward, train our DQN.
	discrim_real_placeholder, discrim_real_label_placeholder, discrim_fake_placeholder, discrim_fake_label_placeholder = env.get_discrim_placeholders()
	real_values, real_labels, fake_values, fake_labels = env.get_discrim_placeholder_values()
	real_loss, fake_loss = env.discrim_loss_tensors()
	d_r_loss, d_f_loss, summary, _, loss = sess.run([real_loss, fake_loss, merged, train_q_network, objective_fn],
                                                        {discrim_real_label_placeholder: real_labels,
                                                         discrim_fake_label_placeholder: fake_labels,
                                                         number_placeholder: numbers_batch,
                                                         pixels_placeholder: pixels_batch,
                                                         action_rewards_placeholder: rewards,
                                                         action_selected_placeholder: actions_selected_batch,
                                                         coordinate_placeholder: coordinates_batch,
                                                         discrim_real_placeholder: real_values,
                                                         discrim_fake_placeholder: fake_values})
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
			q_value_estimates = sess.run([estimated_q_value],
                                                     {number_placeholder: np.array([[state['number']]]),
                                                      pixels_placeholder: np.array([local_pix]),
                                                      coordinate_placeholder: np.array([[state['coordinate']]])
                                                     })
			selected_action = np.argmax(q_value_estimates[0])
			next_state, _, done, _ = env.step(selected_action)
			state = next_state
		print("--------------------")
		print("--------------------")
		print("Test with no randomness")
		env.render()
		print("--------------------")
		print("--------------------")

