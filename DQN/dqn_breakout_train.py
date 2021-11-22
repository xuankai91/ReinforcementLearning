# -*- coding: utf-8 -*-
## credits to: https://www.udemy.com/course/deep-reinforcement-learning-in-python/

import numpy as np
import matplotlib.pyplot as plt
import gym
import ale_py
import sys
from datetime import datetime

# %tensorflow_version 2.x
import tensorflow as tf

from tqdm import tqdm

# setup for Tensorflow V1
tf.compat.v1.disable_eager_execution()

# SETTINGS
MAX_EXPERIENCES = 500000
MIN_EXPERIENCES = 50000
TARGET_UPDATE_PERIOD = 10000
IM_SIZE = 84
K = 4 #env.action_space.n

##########################################################################################################################################################################
##########################################################################################################################################################################

class ImageTransformer:
  def __init__(self):
    with tf.compat.v1.variable_scope("image_transformer"):
      self.input_state = tf.compat.v1.placeholder(tf.uint8, shape=[210, 160, 3])
      self.output = tf.image.rgb_to_grayscale(self.input_state)
      self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
      self.output = tf.image.resize(
        self.output,
        [IM_SIZE, IM_SIZE],
        method='nearest')
      self.output = tf.squeeze(self.output)

  def transform(self, state, sess=None):
    sess = sess or tf.compat.v1.get_default_session()
    return sess.run(self.output, {self.input_state: state})

##########################################################################################################################################################################
##########################################################################################################################################################################

def update_state(state, obs_small):
  '''
  This function is used to modify the observation into a state space that contains some of the preceeding observations.
  '''
  return np.append(state[:,:,1:], np.expand_dims(obs_small, 2), axis=2)

##########################################################################################################################################################################
##########################################################################################################################################################################

class ReplayMemory:
  def __init__(self, size=MAX_EXPERIENCES, frame_height=IM_SIZE, frame_width=IM_SIZE, 
               agent_history_length=4, batch_size=32):
    """
    Args:
        size: Integer, Number of stored transitions
        frame_height: Integer, Height of a frame of an Atari game
        frame_width: Integer, Width of a frame of an Atari game
        agent_history_length: Integer, Number of frames stacked together to create a state
        batch_size: Integer, Number of transitions returned in a minibatch
    """
    self.size = size #max no of experiences to keep in buffer
    self.frame_height = frame_height #height of (transformed image) observation
    self.frame_width = frame_width #width of (transformed image) observation
    self.agent_history_length = agent_history_length # history length to transform the last n observations to a state vector
    self.batch_size = batch_size # network batch size

    self.count = 0 # to keep track of how filled buffer is
    self.current = 0 # to keep track of insertion points
    
    # Pre-allocate memory (actions, rewards, observations/frames, done flag) for the max no. of experiences
    self.actions = np.empty(self.size, dtype=np.int32)
    self.rewards = np.empty(self.size, dtype=np.float32)
    self.frames = np.empty((self.size, self.frame_height, self.frame_width), dtype=np.uint8)
    self.terminal_flags = np.empty(self.size, dtype=np.bool)
    
    # Pre-allocate memory for the states and new_states in a minibatch 
    self.states = np.empty((self.batch_size, self.agent_history_length, 
                            self.frame_height, self.frame_width), dtype=np.uint8)
    
    self.new_states = np.empty((self.batch_size, self.agent_history_length, 
                                self.frame_height, self.frame_width), dtype=np.uint8)
    
    self.indices = np.empty(self.batch_size, dtype=np.int32)

  ## --------------------------------------------------------- ##  
  def add_experience(self, action, frame, reward, terminal):
    """
    Args:
        action: An integer-encoded action
        frame: One grayscale frame of the game
        reward: reward the agend received for performing an action
        terminal: A bool stating whether the episode terminated
    """
    if frame.shape != (self.frame_height, self.frame_width):
      raise ValueError('Dimension of frame is wrong!')
    
    # using self.current as an index
    self.actions[self.current] = action
    self.frames[self.current, ...] = frame #equivalent to [self.current,:,:]
    self.rewards[self.current] = reward
    self.terminal_flags[self.current] = terminal

    # counters are circular
    self.count = max(self.count, self.current+1) # this value maxes out at max episodes
    self.current = (self.current + 1) % self.size # this value resets to 0 after reaching max episodes

  ## --------------------------------------------------------- ##        
  def _get_state(self, index):
    # if there are no played experiences
    if self.count is 0:
      raise ValueError("The replay memory is empty!")
    # if index is less than the length of the history
    if index < self.agent_history_length - 1:
      raise ValueError("Index must be min 3")
    # otherwise, return the last 4 consecutive observations as the state
    return self.frames[index-self.agent_history_length+1:index+1, ...] #indexed as [t-3:t+1]
      
  def _get_valid_indices(self):
    # pick valid indices, the length of a mini-batch, for experience replay
    for i in range(self.batch_size):
      while True:
        # pick a random integer
        index = np.random.randint(low = self.agent_history_length, high = self.count - 1, size = 1)[0]
        
        # if selected index is less than the history length, pick again (may not be necessary?)
        if index < self.agent_history_length:
          continue
        
        # if selected index is in between the last 4 index of the current index, pick again
        if index >= self.current and index - self.agent_history_length <= self.current:
          continue
        
        # if there is any 'done' flag within the last 3 consecutive frames, pick again
        if self.terminal_flags[index - self.agent_history_length:index].any():
          continue
        
        # if selected index passes all the checks, break, 
        break
      
      # append the valid selected index to the list
      self.indices[i] = index

  ## --------------------------------------------------------- ##         
  def get_minibatch(self):
    """
    Returns a minibatch of self.batch_size transitions
    """
    if self.count < self.agent_history_length:
      raise ValueError('Not enough memories to get a minibatch')
    
    # use previous method to fill the valid indices array
    self._get_valid_indices()
        
    for i, idx in enumerate(self.indices):
      # for each valid index, get the previous state and subsequent state
      # this arrangement saves memory space, as one only needs to get 
      # history+1 (for this example, 5) observations to form the previous and next state vectors
      self.states[i] = self._get_state(idx - 1)
      self.new_states[i] = self._get_state(idx)
    
    # return mini-batch of S, A, R, S' and associated done flags
    # transpose rearranges the states to be in the format (num_samples, height, width, "color" depth)
    # which will be used in the tensorflow convolution layer
    return np.transpose(self.states, axes=(0, 2, 3, 1)), self.actions[self.indices], self.rewards[self.indices], np.transpose(self.new_states, axes=(0, 2, 3, 1)), self.terminal_flags[self.indices]

##########################################################################################################################################################################
##########################################################################################################################################################################

class DQN:
  def __init__(self, K, conv_layer_sizes, hidden_layer_sizes, scope):

    self.K = K # no. of actions
    self.scope = scope # tensorflow scope

    with tf.compat.v1.variable_scope(scope): # indicates to tf that every variable inside this block should be within the given scope

      ## inputs and targets

      # tensorflow convolution layer needs the order to be (num_samples, height, width, "color")
      self.X = tf.compat.v1.placeholder(tf.float32, shape=(None, IM_SIZE, IM_SIZE, 4), name='X') 

      self.G = tf.compat.v1.placeholder(tf.float32, shape=(None,), name='G')
      self.actions = tf.compat.v1.placeholder(tf.int32, shape=(None,), name='actions')
      
      ## --------------------------------------------------------- ## 
      ## calculate output and cost
      # convolutional layers
      Z = self.X / 255.0 # normalise color values to be between [0..1]
      for num_output_filters, filtersz, poolsz in conv_layer_sizes:
        Z = tf.compat.v1.layers.conv2d(
          Z, # input
          num_output_filters, # filters
          filtersz, # kernel size
          poolsz, # strides
          activation=tf.nn.relu
        )

      # fully connected layers
      Z = tf.compat.v1.layers.flatten(Z)
      for M in hidden_layer_sizes:
        Z = tf.compat.v1.layers.dense(Z, M)

      # final output layer
      self.predict_op = tf.compat.v1.layers.dense(Z, K)

      selected_action_values = tf.math.reduce_sum(
        self.predict_op * tf.one_hot(self.actions, K),
        axis = 1 #to reduce along rows. deprecated line: `reduction_indices=[1]`
      )

      ## define cost function
      cost = tf.math.reduce_mean(tf.compat.v1.losses.huber_loss(self.G, selected_action_values))
      self.cost = cost

      ## --------------------------------------------------------- ## 
      ## select optimizer
      self.train_op = tf.compat.v1.train.AdamOptimizer(1e-5).minimize(cost)


  ## --------------------------------------------------------- ##     
  def copy_from(self, other):
    mine = [t for t in tf.compat.v1.trainable_variables() if t.name.startswith(self.scope)]
    mine = sorted(mine, key=lambda v: v.name)
    theirs = [t for t in tf.compat.v1.trainable_variables() if t.name.startswith(other.scope)]
    theirs = sorted(theirs, key=lambda v: v.name)

    ops = []
    for p, q in zip(mine, theirs):
      op = p.assign(q)
      ops.append(op)
    self.session.run(ops)

  ## --------------------------------------------------------- ## 
  def save(self,filename = './'):
    params = [t for t in tf.compat.v1.trainable_variables() if t.name.startswith(self.scope)]
    params = self.session.run(params)
    np.savez(filename + 'tf_dqn_weights.npz', *params)

  def load(self,filename = './'):
    params = [t for t in tf.compat.v1.trainable_variables() if t.name.startswith(self.scope)]
    npz = np.load(filename + 'tf_dqn_weights.npz')
    ops = []
    for p, (_, v) in zip(params, npz.iteritems()):
      ops.append(p.assign(v))
    self.session.run(ops)

  ## --------------------------------------------------------- ## 
  def set_session(self, session):
    self.session = session

  def predict(self, states):
    return self.session.run(self.predict_op, feed_dict={self.X: states})

  def update(self, states, actions, targets):
    c, _ = self.session.run(
      [self.cost, self.train_op],
      feed_dict={
        self.X: states,
        self.G: targets,
        self.actions: actions
      }
    )
    # returns the cost
    return c
  
  ## --------------------------------------------------------- ## 
  # epsilon-greedy action selection
  def sample_action(self, x, eps):
    if np.random.random() < eps:
      return np.random.choice(self.K)
    else:
      return np.argmax(self.predict([x])[0])

def learn(model, target_model, experience_replay_buffer, gamma, batch_size):
  # Sample experiences
  states, actions, rewards, next_states, dones = experience_replay_buffer.get_minibatch()

  # Calculate targets
  next_Qs = target_model.predict(next_states)
  next_Q = np.amax(next_Qs, axis=1)
  # get logical_not for done flag, so that there is no discounted next state value if episode has terminated. 
  # i.e. multiply gamma + maxQ by 0 when done = 1, else 1. note that np.invert(dones) is outdated.
  targets = rewards + np.logical_not(dones).astype(np.float32) * gamma * next_Q 

  # Update model
  loss = model.update(states, actions, targets)
  return loss
  
##########################################################################################################################################################################
##########################################################################################################################################################################

# play one episode
def play_one(
  env,
  sess,
  total_t,
  experience_replay_buffer,
  behav_model,
  target_model,
  image_transformer,
  gamma,
  batch_size,
  epsilon,
  epsilon_change,
  epsilon_min):

  t0 = datetime.now()

  # Reset the environment
  obs = env.reset()
  obs_small = image_transformer.transform(obs, sess)
  state = np.stack([obs_small] * 4, axis=2) # repeate initial observation 4 times to make the state
  loss = None
  done = False
  
  total_time_training = 0
  num_steps_in_episode = 0
  episode_reward = 0

  # start playing episode
  while not done:

    # Update target network asynchronously
    if total_t % TARGET_UPDATE_PERIOD == 0:
      target_model.copy_from(behav_model)
      print("Copied model parameters to target network. total_t = %s, period = %s" % (total_t, TARGET_UPDATE_PERIOD))


    # Take action
    action = behav_model.sample_action(state, epsilon)
    
    # get observation, reward, done flag
    obs, reward, done, _ = env.step(action)

    # transform next observation to next state
    obs_small = image_transformer.transform(obs, sess)
    next_state = update_state(state, obs_small)

    # Compute total reward
    episode_reward += reward

    # Save the latest experience
    experience_replay_buffer.add_experience(action, obs_small, reward, done)    

    # Train the model, keep track of time
    t0_2 = datetime.now()
    loss = learn(behav_model, target_model, experience_replay_buffer, gamma, batch_size) # returns the loss for this learning iteration, use target network to predict Q
    dt = datetime.now() - t0_2

    # More debugging info
    total_time_training += dt.total_seconds()
    num_steps_in_episode += 1

    # update state
    state = next_state
    total_t += 1
    
    # fixed epsilon after a certain threshold
    epsilon = max(epsilon - epsilon_change, epsilon_min)

  return total_t, episode_reward, (datetime.now() - t0), num_steps_in_episode, total_time_training/num_steps_in_episode, epsilon

##########################################################################################################################################################################
##########################################################################################################################################################################
## MAIN SCRIPT ##

# reset tensorflow graph
tf.compat.v1.reset_default_graph()

# close previous sessions if exist
if 'sess' in locals():
  if not sess._closed:
    sess.close()
  del sess


# hyperparams and initialize stuff
conv_layer_sizes = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
hidden_layer_sizes = [512]
gamma = 0.99
batch_sz = 32

num_episodes = 3500
total_t = 0
experience_replay_buffer = ReplayMemory()
episode_rewards = np.zeros(num_episodes)

# epsilon
# decays linearly until 0.1
epsilon = 1.0
epsilon_min = 0.1
epsilon_change = (epsilon - epsilon_min) / MAX_EXPERIENCES

# Create environment
env = gym.make("Breakout-v0") #gym.envs.make("Breakout-v0")

# Create models
behav_model = DQN(
  K=K,
  conv_layer_sizes=conv_layer_sizes,
  hidden_layer_sizes=hidden_layer_sizes,
  scope="behav_model")

target_model = DQN(
  K=K,
  conv_layer_sizes=conv_layer_sizes,
  hidden_layer_sizes=hidden_layer_sizes,
  scope="target_model"
)

image_transformer = ImageTransformer()


#
with tf.compat.v1.Session() as sess:
  behav_model.set_session(sess)
  target_model.set_session(sess)
  sess.run(tf.compat.v1.global_variables_initializer())
  
  # populate experience buffer
  print("Populating experience replay buffer...")
  obs = env.reset()
  for i in tqdm(range(MIN_EXPERIENCES)):
    action = np.random.choice(K)
    obs, reward, done, _ = env.step(action)
    obs_small = image_transformer.transform(obs, sess) # not used anymore
    experience_replay_buffer.add_experience(action, obs_small, reward, done)
    # if game ends before minimal number of experiences are needed, reset environment
    if done:
        obs = env.reset()


  # Play a number of episodes and learn!
  t0 = datetime.now()
  for i in tqdm(range(num_episodes)):

    total_t, episode_reward, duration, num_steps_in_episode, time_per_step, epsilon = play_one(
      env,
      sess,
      total_t,
      experience_replay_buffer,
      behav_model,
      target_model,
      image_transformer,
      gamma,
      batch_sz,
      epsilon,
      epsilon_change,
      epsilon_min,
    )
    episode_rewards[i] = episode_reward

    last_100_avg = episode_rewards[max(0, i - 100):i + 1].mean()
    print("Episode:", i,
      "Duration:", duration,
      "Num steps:", num_steps_in_episode,
      "Reward:", episode_reward,
      "Training time per step:", "%.3f" % time_per_step,
      "Avg Reward (Last 100):", "%.3f" % last_100_avg,
      "Epsilon:", "%.3f" % epsilon
    )
    sys.stdout.flush()
  print("Total duration:", datetime.now() - t0)

  behav_model.save()




