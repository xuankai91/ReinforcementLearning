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

# setup for Tensorflow V1
tf.compat.v1.disable_eager_execution()

#
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

  
## --------------------------------------------------------- ## 
# play one episode & save attempt
def save_one(
  env,
  sess,
  behav_model,
  image_transformer,
  attempt,
  epsilon=0
  ):
  
  env = gym.wrappers.Monitor(env,'./play/%d' % attempt,force=True)

  # Reset the environment
  obs = env.reset()
  obs_small = image_transformer.transform(obs, sess)
  state = np.stack([obs_small] * 4, axis=2) # repeate initial observation 4 times to make the state
  loss = None
  done = False
  
  num_steps_in_episode = 0
  episode_reward = 0

  # start playing episode
  while not done:
    # Take action
    action = behav_model.sample_action(state, epsilon)
    
    # get observation, reward, done flag
    obs, reward, done, _ = env.step(action)

    # transform next observation to next state
    obs_small = image_transformer.transform(obs, sess)
    next_state = update_state(state, obs_small)

    # Compute total reward
    episode_reward += reward

    # More debugging info
    num_steps_in_episode += 1

    # update state
    state = next_state
    
  env.close()
  return episode_reward, num_steps_in_episode

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

# Create environment
env = gym.make("Breakout-v0") #gym.envs.make("Breakout-v0")

# Create models
behav_model = DQN(
  K=K,
  conv_layer_sizes=conv_layer_sizes,
  hidden_layer_sizes=hidden_layer_sizes,
  scope="behav_model")

image_transformer = ImageTransformer()

print('model & image transformer instantiated')

#
with tf.compat.v1.Session() as sess:
  behav_model.set_session(sess)
  print('session set')
  sess.run(tf.compat.v1.global_variables_initializer())
  # load trained model
  behav_model.load()
  print('weights loaded')

  
  # play and save video
  for i in range(5):
    _,steps = save_one(env,sess,behav_model,image_transformer,i)
    print('episode %d played, total steps: %d' % (i,steps))






