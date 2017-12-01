import gym
import random

import numpy as np
import tensorflow as tf

from PIL import Image

import sys, time

import os

# Resolution 210 x 160 x 3
# Min reward 0, Max reward 1
# Only 1 reward for every hit object


# Random play for 1000 episodes
# Rewards from 0 to 9
# Mean reward 1.277
# Std 1.307


REPLAY_MEMORY_SIZE = 1000000
EPSILON_MIN = 0.01
EPSILON_MAX = 1.0
NUMBER_OF_EPISODES = 1000000
END_AT_TOTAL_STEPS = 10000000
START_UPDATE_AT = 10000
END_UPDATE_AT = 20000
MAX_STEPS = 10000
LEARNING_RATE = 0.001
MINIBATCH_SIZE = 32
DISCOUNT_FACTOR = 0.99
TARGET_UPDATE_FREQ = 10000
MINIMUM_SAMPLE_SIZE = 10000


RESIZED_IMAGE_SIZE = 32

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def update_epsilon(total_steps):
  if total_steps < START_UPDATE_AT:
    return EPSILON_MAX
  if total_steps > END_UPDATE_AT:
    return EPSILON_MIN
  return EPSILON_MAX + (EPSILON_MIN - EPSILON_MAX) / (END_UPDATE_AT - START_UPDATE_AT) * (total_steps - START_UPDATE_AT)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def leakyRelu(input):
  return tf.maximum(input, 0.1 * input)

def train(config, summary_dir):
  print("Inside the train function")
  # Create session
  session = tf.InteractiveSession()

  exploration = False
  initialization = False
  parameteralter = False

  if config == "egreedy":
    exploration = True
  elif config == "posinit":
    initialization = True
  elif config == "egreedyposinit":
    exploration = True
    initialization = True
  elif config == "paramalter":
    parameteralter = True
  else:
    print("No such configuration")
    return


  # TODO: Load if there is a model present

  # Create environment
  environment = gym.make("Breakout-v0")
  input_shape = environment.observation_space.shape
  output_size = environment.action_space.n

  print("environment made")

  with tf.name_scope("image_prep"):
    input_image = tf.placeholder(tf.float32, input_shape)
    gray_image = tf.image.rgb_to_grayscale(input_image)
    resized_image = tf.image.resize_images(gray_image, [RESIZED_IMAGE_SIZE, RESIZED_IMAGE_SIZE], method=tf.image.ResizeMethod.AREA)
    normalized_image = tf.squeeze(resized_image)
    normalized_image = normalized_image / 256.0

    # TODO: figure out why it isn't working and expects a 210, 160, 3 image
    # tf.summary.image("normalized_image", [resized_image], 10)

  print("imageprep made")

  # TODO: parametrize sizes
  with tf.name_scope("input"):
    input_state = tf.placeholder(tf.float32, [None, RESIZED_IMAGE_SIZE, RESIZED_IMAGE_SIZE, 4], name="input_state")
    #norm_state = tf.contrib.layers.layer_norm(input_state)
    tf.summary.histogram("input_state", input_state)
    #tf.summary.histogram("norm_state", norm_state)
  
  print("input made")

  with tf.name_scope("target"):
    targetQ = tf.placeholder(tf.float32, [None], name="targetQ")
    targetActionMask = tf.placeholder(tf.float32, [None, output_size], name="targetActionMask")
  
  print("target made")

  initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False, seed=None, dtype=tf.float32)

  print("initializer made")

  with tf.name_scope("conv1"):
    W_conv1 = tf.get_variable("conv1/W_conv1", shape=[3, 3, 4, 8], initializer=initializer)
    b_conv1 = tf.get_variable("conv1/b_conv1", shape=[8], initializer=initializer)
    r_conv1 = conv2d(input_state, W_conv1, 2) + b_conv1
    #n_conv1 = tf.contrib.layers.layer_norm(r_conv1)
    h_conv1 = leakyRelu(r_conv1)

    tf.summary.histogram("W_conv1", W_conv1)
    tf.summary.histogram("b_conv1", b_conv1)
    tf.summary.histogram("r_conv1", r_conv1)
    #tf.summary.histogram("n_conv1", n_conv1)
    tf.summary.histogram("h_conv1", h_conv1)
  
  with tf.name_scope("conv2"):
    W_conv2 = tf.get_variable("conv2/W_conv2", shape=[3, 3, 8, 16], initializer=initializer)
    b_conv2 = tf.get_variable("conv2/b_conv2", shape=[16], initializer=initializer)
    r_conv2 = conv2d(h_conv1, W_conv2, 2) + b_conv2
    #n_conv2 = tf.contrib.layers.layer_norm(r_conv2)
    h_conv2 = leakyRelu(r_conv2)

    tf.summary.histogram("W_conv2", W_conv2)
    tf.summary.histogram("b_conv2", b_conv2)
    tf.summary.histogram("r_conv2", r_conv2)
    #tf.summary.histogram("n_conv2", n_conv2)
    tf.summary.histogram("h_conv2", h_conv2)
  
  with tf.name_scope("conv3"):
    W_conv3 = tf.get_variable("conv3/W_conv3", shape=[3, 3, 16, 32], initializer=initializer)
    b_conv3 = tf.get_variable("conv3/b_conv3", shape=[32], initializer=initializer)
    r_conv3 = conv2d(h_conv2, W_conv3, 2) + b_conv3
    #n_conv3  = tf.contrib.layers.layer_norm(r_conv3)
    h_conv3 = leakyRelu(r_conv3)

    tf.summary.histogram("W_conv3", W_conv3)
    tf.summary.histogram("b_conv3", b_conv3)
    tf.summary.histogram("r_conv3", r_conv3)
    #tf.summary.histogram("n_conv3", n_conv3)
    tf.summary.histogram("h_conv3", h_conv3)

  with tf.name_scope("flatten"):
    flat_conv3 = tf.reshape(h_conv3, [-1, 512])
    tf.summary.histogram("flat_conv3", flat_conv3)

  with tf.name_scope("hidden"):
    W1 = tf.get_variable("hidden/W1", shape=[512, 128], initializer=initializer)
    b1 = tf.get_variable("hidden/b1", shape=[128], initializer=initializer)
    r_hidden = tf.matmul(flat_conv3, W1) + b1
    #n_hidden = tf.contrib.layers.layer_norm(r_hidden)
    h_hidden = leakyRelu(r_hidden)

    tf.summary.histogram("W1", W1)
    tf.summary.histogram("b1", b1)
    tf.summary.histogram("r_hidden", r_hidden)
    #tf.summary.histogram("n_hidden", n_hidden)
    tf.summary.histogram("h_hidden", h_hidden)

  with tf.name_scope("output"):
    W2 = tf.get_variable("output/W2", shape=[128, output_size], initializer=initializer)
    b2 = tf.get_variable("output/b2", shape=[output_size], initializer=initializer)
    Q = tf.matmul(h_hidden, W2) + b2

    tf.summary.histogram("W2", W2)
    tf.summary.histogram("b2", b2)
    tf.summary.histogram("Q", Q)

  weights = [W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W1, b1, W2, b2]
  
  with tf.name_scope("Q_learning"):
    maskedQ = tf.reduce_sum(tf.multiply(Q, targetActionMask), reduction_indices=[1], name="maskedQ")
    loss = tf.reduce_mean(tf.square(tf.subtract(maskedQ, targetQ)), name="loss")

    tf.summary.scalar("loss", loss)

  with tf.name_scope("train"):
    train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

  print("Network made")

  # Merge summaries
  merged_summary = tf.summary.merge_all()
  tf.global_variables_initializer().run()

  # Summary writer and model saver
  summary_writer = tf.summary.FileWriter(summary_dir, session.graph)
  model_saver = tf.train.Saver()

  # TODO: if there is a model load it
  model_saver.restore(session, "/tmp/tensorboard/model-0.ckpt")
  print("Model restored")

  epsilon = EPSILON_MAX
  replay_memory = []
  image_array = []
  rewards_array = []

  total_steps = 0
  fixed_weights = session.run(weights)
  
  # Run episodes
  for episode in range(NUMBER_OF_EPISODES):
    print(episode)
    rewards = 0
    # Reset environment
    image = environment.reset()
    norm_image = normalized_image.eval(feed_dict = {input_image: image})
    image_array = [norm_image, norm_image, norm_image, norm_image]
    state = np.stack(image_array, axis=2)

    # Take steps
    for step in range(MAX_STEPS):
      environment.render()
      # Pick the next action and execute it
      action = None
      epsilon = 0.0
      if random.random() < epsilon:
        action = environment.action_space.sample()
      else:
        q_values = session.run(Q, feed_dict={input_state: [state]})
        print(q_values)
        raw_input("Press Enter to continue...")
        action = q_values.argmax()
        print(action)

      image, reward, done, _ = environment.step(action)
      rewards += reward
      norm_image = normalized_image.eval(feed_dict = {input_image: image})
      image_array.append(norm_image)
      image_array.pop(0)
      observation = np.stack(image_array, axis=2)

      state = observation

      if done:
        break
    
    rewards_array.append(rewards)
    if len(rewards_array) > 100:
      rewards_array.pop(0)
    print("Epsilon: {}, Minimum: {}, Mean: {}, Maximum: {}".format(epsilon, min(rewards_array), np.mean(rewards_array), max(rewards_array)))

def play_test():
  env = gym.make("Breakout-v0")
  image = env.reset()
  norm_image = normalized_image.eval(feed_dict = {input_image: image})
  image_array = [norm_image, norm_image, norm_image, norm_image]
  state = np.stack(image_array, axis=2)
  done = False
  steps = 0
  for i in range(10000):
    env.render()
    state = np.stack(image_array, axis=2)
    q_values = Q.eval(feed_dict = {input_state: [state]})
    _, _, done, _ = env.step(np.argmax(q_values))
    if done:
      break
    
  return 

def play_random():
  env = gym.make("Breakout-v0")
  done = False
  rewards_array = []
  total_steps = 0
  for episode in range(10000):
    rewards = 0
    env.reset()
    for step in range(100000):
      #env.render()
      _, reward, done, _ = env.step(env.action_space.sample())
      total_steps += 1
      rewards += reward
      if done:
        break
    
    rewards_array.append(rewards)
    print(total_steps)
  
  print("Minimum: %s" % min(rewards_array))
  print("Mean: %s" % np.mean(rewards_array))
  print("Maximum: %s" % max(rewards_array))

  return

if __name__ == "__main__":
  train(sys.argv[1], "/tmp/tensorboard")