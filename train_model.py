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

REPLAY_MEMORY_SIZE = 100000
EPSILON_MIN = 0.1
EPSILON_MAX = 1.0
NUMBER_OF_EPISODES = 1000000
END_AT_TOTAL_STEPS = 2500000
START_UPDATE_AT = 10000
END_UPDATE_AT = 250000
MAX_STEPS = 10000
LEARNING_RATE = 0.001
MINIBATCH_SIZE = 32
DISCOUNT_FACTOR = 0.99
TARGET_UPDATE_FREQ = 10000
MINIMUM_SAMPLE_SIZE = 10000


RESIZED_IMAGE_SIZE = 32

# Updates the epsilon exploration parameter gradually from EPSILON_MAX to EPSILON_MIN
def update_epsilon(total_steps):
  if total_steps < START_UPDATE_AT:
    return EPSILON_MAX
  if total_steps > END_UPDATE_AT:
    return EPSILON_MIN
  return EPSILON_MAX + (EPSILON_MIN - EPSILON_MAX) / (END_UPDATE_AT - START_UPDATE_AT) * (total_steps - START_UPDATE_AT)

# Creates a 2D convolution layer with input x, variables W, and stride
def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

# Creates a leaky ReLU layer with given input
def leakyRelu(input):
  return tf.maximum(input, 0.1 * input)

# Creates and trains the reinforcement learning model, saves the model to model_dir
def train(model_dir):
  # Create Tensorflow session
  session = tf.InteractiveSession()

  # Create environment
  environment = gym.make("Breakout-v0")
  input_shape = environment.observation_space.shape
  output_size = environment.action_space.n

  # image_prep takes the input images and outputs square monochrome pictures normalized to the [0,1] interval
  with tf.name_scope("image_prep"):
    input_image = tf.placeholder(tf.float32, input_shape)
    gray_image = tf.image.rgb_to_grayscale(input_image)
    resized_image = tf.image.resize_images(gray_image, [RESIZED_IMAGE_SIZE, RESIZED_IMAGE_SIZE], method=tf.image.ResizeMethod.AREA)
    normalized_image = tf.squeeze(resized_image)
    normalized_image = normalized_image / 256.0

  # the input layer for the DQN agent
  with tf.name_scope("input"):
    input_state = tf.placeholder(tf.float32, [None, RESIZED_IMAGE_SIZE, RESIZED_IMAGE_SIZE, 4], name="input_state")
    tf.summary.histogram("input_state", input_state)

  # the placeholder for the target Q values to give a basis to train the model
  with tf.name_scope("target"):
    targetQ = tf.placeholder(tf.float32, [None], name="targetQ")
    targetActionMask = tf.placeholder(tf.float32, [None, output_size], name="targetActionMask")

  # Xavier initializer to prevent gradient vanishing or exploding
  initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False, seed=None, dtype=tf.float32)

  # The first convolutional layer input: [32x32x4] output: [16x16x8] filter: [3x3x4x8] stride: 2
  with tf.name_scope("conv1"):
    W_conv1 = tf.get_variable("conv1/W_conv1", shape=[3, 3, 4, 8], initializer=initializer)
    b_conv1 = tf.get_variable("conv1/b_conv1", shape=[8], initializer=initializer)
    r_conv1 = conv2d(input_state, W_conv1, 2) + b_conv1
    h_conv1 = leakyRelu(r_conv1)

    tf.summary.histogram("W_conv1", W_conv1)
    tf.summary.histogram("b_conv1", b_conv1)
    tf.summary.histogram("r_conv1", r_conv1)
    tf.summary.histogram("h_conv1", h_conv1)
  
  # The second convolutional layer input: [16x16x8] output: [8x8x16] filter: [3x3x8x16] stride: 2
  with tf.name_scope("conv2"):
    W_conv2 = tf.get_variable("conv2/W_conv2", shape=[3, 3, 8, 16], initializer=initializer)
    b_conv2 = tf.get_variable("conv2/b_conv2", shape=[16], initializer=initializer)
    r_conv2 = conv2d(h_conv1, W_conv2, 2) + b_conv2
    h_conv2 = leakyRelu(r_conv2)

    tf.summary.histogram("W_conv2", W_conv2)
    tf.summary.histogram("b_conv2", b_conv2)
    tf.summary.histogram("r_conv2", r_conv2)
    tf.summary.histogram("h_conv2", h_conv2)
  
  # The third convolutional layer input: [8x8x16] output: [4x4x32] filter: [3x3x16x32] stride: 2
  with tf.name_scope("conv3"):
    W_conv3 = tf.get_variable("conv3/W_conv3", shape=[3, 3, 16, 32], initializer=initializer)
    b_conv3 = tf.get_variable("conv3/b_conv3", shape=[32], initializer=initializer)
    r_conv3 = conv2d(h_conv2, W_conv3, 2) + b_conv3
    h_conv3 = leakyRelu(r_conv3)

    tf.summary.histogram("W_conv3", W_conv3)
    tf.summary.histogram("b_conv3", b_conv3)
    tf.summary.histogram("r_conv3", r_conv3)
    tf.summary.histogram("h_conv3", h_conv3)

  # Flatten the output tensor to a series of vectors
  with tf.name_scope("flatten"):
    flat_conv3 = tf.reshape(h_conv3, [-1, 512])
    tf.summary.histogram("flat_conv3", flat_conv3)

  # The first hidden layer input: 512 output: 128
  with tf.name_scope("hidden"):
    W1 = tf.get_variable("hidden/W1", shape=[512, 128], initializer=initializer)
    b1 = tf.get_variable("hidden/b1", shape=[128], initializer=initializer)
    r_hidden = tf.matmul(flat_conv3, W1) + b1
    h_hidden = leakyRelu(r_hidden)

    tf.summary.histogram("W1", W1)
    tf.summary.histogram("b1", b1)
    tf.summary.histogram("r_hidden", r_hidden)
    tf.summary.histogram("h_hidden", h_hidden)

  # The output layer input: 128 output: 4
  with tf.name_scope("output"):
    W2 = tf.get_variable("output/W2", shape=[128, output_size], initializer=initializer)
    b2 = tf.get_variable("output/b2", shape=[output_size], initializer=initializer)
    Q = tf.matmul(h_hidden, W2) + b2

    tf.summary.histogram("W2", W2)
    tf.summary.histogram("b2", b2)
    tf.summary.histogram("Q", Q)

  # Array containing all the weight references
  weights = [W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W1, b1, W2, b2]
  
  # The loss calculated from the network's Q values and the target Q values
  with tf.name_scope("Q_learning"):
    maskedQ = tf.reduce_sum(tf.multiply(Q, targetActionMask), reduction_indices=[1], name="maskedQ")
    loss = tf.reduce_mean(tf.square(tf.subtract(maskedQ, targetQ)), name="loss")

    tf.summary.scalar("loss", loss)

  # The training operator to minimize loss
  with tf.name_scope("train"):
    train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

  # Merge summaries
  merged_summary = tf.summary.merge_all()
  tf.global_variables_initializer().run()

  # Summary writer and model saver
  summary_writer = tf.summary.FileWriter(".", session.graph)
  model_saver = tf.train.Saver()

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
      # Pick the next action and execute it
      action = None
      epsilon = update_epsilon(total_steps)
      if random.random() < epsilon:
        # Random action
        action = environment.action_space.sample()
      else:
        # Greedy action
        q_values = session.run(Q, feed_dict={input_state: [state]})
        action = q_values.argmax()

      # Stack together the last 4 images
      image, reward, done, _ = environment.step(action)
      rewards += reward
      norm_image = normalized_image.eval(feed_dict = {input_image: image})
      image_array.append(norm_image)
      image_array.pop(0)
      observation = np.stack(image_array, axis=2)

      total_steps += 1
      if total_steps % 1000 == 0:
        print(total_steps)

      # Update replay memory
      replay_memory.append((state, action, reward, observation, done))
      if total_steps > REPLAY_MEMORY_SIZE:
        replay_memory.pop(0)
      
      state = observation

      # Sample a random minibatch
      if total_steps > MINIMUM_SAMPLE_SIZE:
        minibatch = random.sample(replay_memory, MINIBATCH_SIZE)
        next_states = [m[3] for m in minibatch]
        feed_dict = {input_state: next_states}
        feed_dict.update(zip(weights, fixed_weights))
        q_values = session.run(Q, feed_dict=feed_dict)
        max_q_values = q_values.max(axis=1)

        # Compute target Q values
        # TODO: Vectorize computations
        target_q = np.zeros(MINIBATCH_SIZE)
        target_action_mask = np.zeros((MINIBATCH_SIZE, output_size), dtype=int)
        for i in range(MINIBATCH_SIZE):
          _, action, reward, _, terminal = minibatch[i]
          target_q[i] = reward
          # TODO: uncomment
          if not terminal:
            target_q[i] += DISCOUNT_FACTOR * max_q_values[i]
          target_action_mask[i][action] = 1

        # if total_steps < 11000:
        #   target_q = [1.0] * 32 
        #   fixed_weights = session.run(weights)

        # Gradient descent
        states = [m[0] for m in minibatch]
        feed_dict = {
          input_state: states, 
          targetQ: target_q,
          targetActionMask: target_action_mask,
        }
        _, summary = session.run([train_op, merged_summary], feed_dict=feed_dict)

        # Write summary for TensorBoard
        if total_steps % 1000 == 0:
          summary_writer.add_summary(summary, total_steps)

        # Update target network
        if total_steps % TARGET_UPDATE_FREQ == 0:
          fixed_weights = session.run(weights)
        
        # Save model
        if total_steps % 10000 == 0:
          save_path = model_saver.save(session, model_dir + "/model-" + str(int(np.mean(rewards_array))) + ".ckpt")
          print("Model saved in file: %s" % save_path)

      if done:
        break
    
    rewards_array.append(rewards)
    if len(rewards_array) > 100:
      rewards_array.pop(0)
    print("Epsilon: {}, Mean: {}, Std: {}, Minimum: {}, Maximum: {}".format(epsilon, np.mean(rewards_array), np.std(rewards_array), min(rewards_array), max(rewards_array)))

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
  train(sys.argv[1])