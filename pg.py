from config import *
from utils import *
import os
import gym
import drawing_env
from drawing_env.envs.ops import *
import numpy as np
import tensorflow as tf

# Policy gradient architecture. Return logits for each action's probability
# TODO: Try different architecture?
def policy_network(pixels, coordinate, number, num_actions_per_pixel):
  batch_size = tf.shape(pixels)[0]
  reshaped_input = tf.reshape(pixels, tf.stack([batch_size, LOCAL_DIMENSION, LOCAL_DIMENSION, 1]))
    
  h0 = lrelu(conv2d(reshaped_input, 4, 2, 2, 1, 1, name="conv1"))
  h1 = lrelu(conv2d(h0, 8, 2, 2, 1, 1, name="conv2"))
  h2 = lrelu(conv2d(h1, 16, 2, 2, 1, 1, name="conv3"))
  h2_flatted = tf.layers.flatten(h2)
    
  # Append coordinate and number label to last layer because we don't want it to be convolved with
  # the pixel values.
  coord_as_rank2 = tf.expand_dims(coordinate, 1)
  number_as_rank2 = tf.expand_dims(number, 1)
  h3 = tf.contrib.layers.fully_connected(tf.concat([h2_flatted, coord_as_rank2, number_as_rank2], axis=1),
                                         FULL_DIMENSION*FULL_DIMENSION)
  h4 = tf.contrib.layers.fully_connected(h3, FULL_DIMENSION)
  output = tf.contrib.layers.fully_connected(h4, num_actions_per_pixel, activation_fn=None)
  return output


class DrawPG(object):
  """
  Abstract Class for implementing a Policy Gradient Based Algorithm
  """
  # TODO: instantiate with digit, remove numbers placeholder?
  # TODO: Remove coordinate placeholder?
  def __init__(self, digit, env, output_path, model_path, log_path, gamma=1, lr=PG_LR,
               use_baseline=True, normalize_advantage=True, batch_size=PG_BATCH_NSTEPS,
               num_batches=PG_NUM_BATCHES, summary_freq=PG_SUMMARY_FREQ, draw_freq=PG_DRAW_FREQ):
    """
    Initialize Policy Gradient Class
    """
    # directory for training outputs
    if not os.path.exists(output_path):
      os.makedirs(output_path)
    if not os.path.exists(model_path):
      os.makedirs(model_path)
            
    # store hyper-params
    self.digit = digit
    self.env = env

    self.output_path = output_path
    self.model_path = model_path
    self.logger = get_logger(log_path)
    self.gamma = gamma
    self.lr = lr
    self.use_baseline = use_baseline
    self.normalize_advantage = normalize_advantage
    self.batch_size = batch_size
    self.num_batches = num_batches
    self.summary_freq = summary_freq
    self.draw_freq = draw_freq

    # action dim
    self.action_dim = self.env.action_space.n
  
    # build model
    self.build()

    
  def add_placeholders_op(self):
    """
    Adds placeholders to the graph
    Set up the observation, action, and advantage placeholder
    """
    self.pixels_placeholder = tf.placeholder(tf.float32, shape=(None, LOCAL_DIMENSION*LOCAL_DIMENSION),
                                             name='pixel_window')
    self.coordinate_placeholder = tf.placeholder(tf.float32, shape=(None,), name='current_coordinate')
    self.number_placeholder = tf.placeholder(tf.float32, shape=(None,), name='digit')
    
    self.taken_action_placeholder = tf.placeholder(tf.int32, shape=(None,), name='taken_action')
    self.advantage_placeholder = tf.placeholder(tf.float32, shape=(None,), name='advantage')
  
  
  def build_policy_network_op(self, scope="policy_network"):
    """
    Build the policy network, construct the tensorflow operation to sample 
    actions from the policy network outputs, and compute the log probabilities
    of the taken actions (for computing the loss later). These operations are 
    stored in self.sampled_action and self.logprob. Must handle both settings
    of self.discrete.
    """
    with tf.variable_scope(scope):
      action_logits = policy_network(self.pixels_placeholder, self.coordinate_placeholder, self.number_placeholder,
                                     self.action_dim)
      self.sampled_action = tf.squeeze(tf.multinomial(action_logits, 1), name='sampled_action_discrete')
      self.logprob = -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.taken_action_placeholder,
                      logits=action_logits, name='taken_action_logprob_discrete')            
  
  
  def add_loss_op(self):
    """
    Sets the loss of a batch, the loss is a scalar 
    """
    # REINFORCE update uses mean over all trajectories of sum over each trajectory of log pi * A_t
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
    with tf.variable_scope(scope):
      self.baseline = tf.squeeze(policy_network(self.pixels_placeholder, self.coordinate_placeholder,
                                                self.number_placeholder, 1), axis=1)
      self.baseline_target_placeholder = tf.placeholder(tf.float32, shape=(None,), name='baseline_target')
      loss = tf.losses.mean_squared_error(self.baseline_target_placeholder, self.baseline)
      self.update_baseline_op = tf.train.AdamOptimizer().minimize(loss, name='calibrate_baseline')

  
  def build(self):
    """
    Build model by adding all necessary variables to tensorflow graph
    """
    # add placeholders
    self.add_placeholders_op()
    # create policy net
    self.build_policy_network_op()
    # add square loss
    self.add_loss_op()
    # add optimizer for the main networks
    self.add_optimizer_op()
  
    if self.use_baseline:
      self.add_baseline_op()


  def initialize(self):
    """
    Assumes the graph has been constructed (have called self.build())
    Creates a tf Session and run initializer of variables

    You don't have to change or use anything here.
    """
    # create tf session, send info to env
    self.sess = tf.Session()
    self.env.set_session(self.sess)
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
    self.dscr_real_loss_placeholder = tf.placeholder(tf.float32, shape=(), name='disc_real_data_loss')
    self.dscr_gen_loss_placeholder = tf.placeholder(tf.float32, shape=(), name='disc_gen_data_loss')
  
    # extra summaries from python -> placeholders
    summaries = [
      tf.summary.scalar("Avg_Reward", self.avg_reward_placeholder),
      tf.summary.scalar("Max_Reward", self.max_reward_placeholder),
      tf.summary.scalar("Std_Reward", self.std_reward_placeholder),
      tf.summary.scalar("Eval_Reward", self.eval_reward_placeholder),
      tf.summary.scalar('Discriminator_Loss_on_Real_Data', self.dscr_real_loss_placeholder),
      tf.summary.scalar('Discriminator_Loss_on_Generated_Data', self.dscr_gen_loss_placeholder),
    ]
            
    # logging
    self.merged = tf.summary.merge(summaries)
    self.file_writer = tf.summary.FileWriter(self.output_path, self.sess.graph) 

    
  def init_statistics(self):
    """
    Defines extra attributes for tensorboard.
    """
    self.avg_reward = 0.
    self.max_reward = 0.
    self.std_reward = 0.
    self.eval_reward = 0.
    self.dscr_real_loss = 0.
    self.dscr_gen_loss = 0.
  

  def update_statistics(self, rewards, scores_eval, d_r_loss, d_g_loss):
    """
    Update the statistics.
  
    Args:
            rewards: deque
            scores_eval: list
    """
    self.avg_reward = np.mean(rewards)
    self.max_reward = np.max(rewards)
    self.std_reward = np.sqrt(np.var(rewards) / len(rewards))
  
    if len(scores_eval) > 0:
      self.eval_reward = scores_eval[-1]

    self.dscr_real_loss = d_r_loss
    self.dscr_gen_loss = d_g_loss
  
  
  def record_summary(self, t):
    """
    Add summary to tfboard
    """
  
    fd = {
      self.avg_reward_placeholder: self.avg_reward, 
      self.max_reward_placeholder: self.max_reward, 
      self.std_reward_placeholder: self.std_reward, 
      self.eval_reward_placeholder: self.eval_reward,
      self.dscr_real_loss_placeholder: self.dscr_real_loss,
      self.dscr_gen_loss_placeholder: self.dscr_gen_loss,
    }
    summary = self.sess.run(self.merged, feed_dict=fd)
    self.file_writer.add_summary(summary, t)
    self.file_writer.flush()

  # TODO: Is the discriminator a general discriminator? How does this initialization work?
  def init_discriminator(self):
    ''' 
    Initialize the DrawEnv's discriminator with some training. 
    '''
    real_prob = 0.5
    fake_prob = 0.5
    for _ in range(5000):
      fake_prob, real_prob = self.env.train_disc_random_fake()
      if real_prob >= 0.75 or fake_prob <= 0.25:
        break

      
  def sample_path(self, env, num_episodes = None):
    """
    Sample path for the environment.
  
    Args:
            num_episodes:   the number of episodes to be sampled 
              if none, sample one batch 
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
  
    while (num_episodes or t < self.batch_size):
      state = env.reset_to_selected_digit(self.digit)      
      pixels, coords, numbers, actions, rewards = [], [], [], [], []
      episode_reward = 0

      # fill out a full image
      for step in range(FULL_DIMENSION*FULL_DIMENSION):
        full_image, crd, nm = state['pixels'], state['coordinate'], state['number']
#        px = full_image
        px = get_local_pixels(full_image, crd, window_size=LOCAL_DIMENSION)
        pixels.append(px)
        coords.append(crd)
        numbers.append(nm)

        # TODO: Ensure this is rank-0
        action = self.sess.run(self.sampled_action, feed_dict={self.pixels_placeholder: px[None,:],
                                                               self.coordinate_placeholder: [crd],
                                                               self.number_placeholder: [nm]})
        state, reward, done, info = env.step(action) # TODO: try stepping with sparse rewards
        actions.append(action)
        rewards.append(reward)
        episode_reward += reward
        
        t += 1
        if (done or step == FULL_DIMENSION*FULL_DIMENSION - 1):
          episode_rewards.append(episode_reward) 
          break
        if (not num_episodes) and t == self.batch_size:
          break
  
      path = {'pixels': np.array(pixels),
              'coords': np.array(coords),
              'numbers': np.array(numbers),
              'actions': np.array(actions),
              'reward': np.array(rewards)}
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
    of Q^pi (s_t, a_t):
    
       G_t = r_t + gamma r_{t+1} + gamma^2 r_{t+2} + ... + gamma^{T-t} r_T
    
    where T is the last timestep of the episode.
    """

    all_returns = []
    for path in paths:
      rewards = path["reward"]
      returns = []
      for r in rewards[::-1]: # G[t-1] = r + gamma G[t], where G[t+1] = 0. Iterate in reverse over rewards.
        returns.append(r + self.gamma * (0 if not returns else returns[-1])) 
      returns = returns[::-1]
      all_returns.append(returns)
    returns = np.concatenate(all_returns)
  
    return returns
  
  
  def calculate_advantage(self, returns, pixels, coordinates, numbers):
    """
    Calculate the advantage
    Args:
            returns: all discounted future returns for each step
            observations: observations
              Calculate the advantages, using baseline adjustment if necessary,
              and normalizing the advantages if necessary.
              If neither of these options are True, just return returns.

    If self.use_baseline = False and self.normalize_advantage = False,
    then the "advantage" is just going to be the returns (and not actually
    an advantage). 

    if self.normalize_advantage:
      after doing the above, normalize the advantages so that they have a mean of 0
      and standard deviation of 1.
    """
    adv = returns
    if self.use_baseline:
      bl = self.sess.run(self.baseline, {self.pixels_placeholder: pixels,
                                         self.coordinate_placeholder: coordinates,
                                         self.number_placeholder: numbers})
      adv = returns - bl 
    if self.normalize_advantage:
      adv = (adv - np.mean(adv))/np.std(adv)
      
    return adv
  
  
  def update_baseline(self, returns, pixels, coordinates, numbers):
    """
    Update the baseline
    """
    self.sess.run(self.update_baseline_op, {self.pixels_placeholder: pixels,
                                            self.coordinate_placeholder: coordinates,
                                            self.number_placeholder: numbers,
                                            self.baseline_target_placeholder: returns})
  
  
  def train(self):
    """
    Performs training
    """
    last_eval = 0 
    last_record = 0
    scores_eval = []

    self.init_discriminator()
    self.init_statistics()
    scores_eval = [] # list of scores computed at iteration time
  
    for t in range(self.num_batches):
      # collect a minibatch of samples
      paths, total_rewards = self.sample_path(self.env) 
      scores_eval = scores_eval + total_rewards
      pixels = np.concatenate([path['pixels'] for path in paths])
      coords = np.concatenate([path['coords'] for path in paths])
      numbers = np.concatenate([path['numbers'] for path in paths])
      actions = np.concatenate([path["actions"] for path in paths])
      rewards = np.concatenate([path["reward"] for path in paths])
      # compute Q-val estimates (discounted future returns) for each time step
      returns = self.get_returns(paths)      
      advantages = self.calculate_advantage(returns, pixels, coords, numbers)

      # run training operation for generator
      if self.use_baseline:
        self.update_baseline(returns, pixels, coords, numbers)
      self.sess.run(self.train_op, feed_dict={
                    self.pixels_placeholder: pixels,
                    self.coordinate_placeholder: coords,
                    self.number_placeholder: numbers,
                    self.taken_action_placeholder: actions, 
                    self.advantage_placeholder: advantages})

      # get losses for discriminator
      dscr_placeholders = env.get_discrim_placeholders()
      dscr_values = env.get_discrim_placeholder_values() 
      dscr_real_loss_tensor, dscr_gen_loss_tensor = env.discrim_loss_tensors()
      dscr_fd = dict(zip(dscr_placeholders, dscr_values))
      d_r_loss, d_g_loss = self.sess.run([dscr_real_loss_tensor, dscr_gen_loss_tensor],
                                         feed_dict=dscr_fd)
              
      # tf stuff: record summary and update saved model weights
      if (t % self.summary_freq == 0):
        self.update_statistics(total_rewards, scores_eval, d_r_loss, d_g_loss)
        self.record_summary(t)
      if (t % self.draw_freq == 0):
        # Also render
        print('Rendering last generated digit...')
        self.env.render()

      # compute reward statistics for this batch and log
      avg_reward = np.mean(total_rewards)
      sigma_reward = np.sqrt(np.var(total_rewards) / len(total_rewards))
      msg = 'Average episode reward for generator | Discriminator loss on real | Discriminator loss on generated: '\
            + '{:04.2f} +/- {:04.2f} | {:04.2f} | {:04.2f}'.format(avg_reward, sigma_reward, d_r_loss, d_g_loss)
      self.logger.info(msg)
    
    self.logger.info("- Training done.")
    
    # save model params
    self.saver = tf.train.Saver()
    self.saver.save(self.sess, self.model_path)


  
  def run(self):
    """
    Apply procedures of training for a PG.
    """
    # initialize
    self.initialize()
    # train model
    self.train()


########### SETUP ##########
# TODO: setup one agent per digit?
digit = 5
env = gym.make('DrawEnv-v0')
output_path = 'tensorboard/pg/'
model_path = output_path + 'weights/'
log_path = output_path + 'logs.txt'

model = DrawPG(digit, env, output_path, model_path, log_path)
model.run()


