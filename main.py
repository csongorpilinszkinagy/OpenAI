import gym
import random

import numpy as np
import tensorflow as tf

REPLAY_MEMORY_SIZE = 1000000
EPSILON_MIN = 0.1
EPSILON_MAX = 1.0
HIDDEN_SIZE_1 = 32
HIDDEN_SIZE_2 = 32
NUMBER_OF_EPISODES = 100000
MAX_STEPS = 1000
LEARNING_RATE = 0.00025
MINIBATCH_SIZE = 32
DISCOUNT_FACTOR = 0.99
TARGET_UPDATE_FREQ = 10000
REG_FACTOR = 0.001
LOG_DIR = "/tmp/dqn"
DROPOUT = 0.9
START_UPDATE_AT = 10000
END_UPDATE_AT = 100000
END_AT_TOTAL_STEPS = 2000000
MINIMUM_SAMPLE_SIZE = 10000

# Create session
session = tf.InteractiveSession()

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_variable(shape):
  return tf.Variable(tf.constant(0.1, shape=shape))

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def variable_summaries(var):
  with tf.name_scope("summaries"):
    mean = tf.reduce_mean(var)
    tf.summary.scalar("mean", mean)
    with tf.name_scope("stddev"):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
    tf.summary.scalar("stddev", stddev)
    tf.summary.scalar("min", tf.reduce_min(var))
    tf.summary.scalar("max", tf.reduce_max(var))
    tf.summary.histogram("histogram", var)

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
  with tf.name_scope(layer_name):
    with tf.name_scope("weights"):
      weights = weight_variable([input_dim, output_dim])
      variable_summaries(weights)
    with tf.name_scope("biases"):
      biases = bias_variable([output_dim])
      variable_summaries(biases)
    with tf.name_scope("Wx_plus_b"):
      preactivate = tf.matmul(input_tensor, weights) + biases
      tf.summary.histogram("pre_activations", preactivate)
    activations = act(preactivate, name="activation")
    tf.summary.histogram("activations", activations)
    return activations, weights, biases

def update_epsilon(total_steps):
  if total_steps < START_UPDATE_AT:
    return EPSILON_MAX
  if total_steps > END_UPDATE_AT:
    return EPSILON_MIN
  return EPSILON_MAX + (EPSILON_MIN - EPSILON_MAX) / (END_UPDATE_AT - START_UPDATE_AT) * (total_steps - START_UPDATE_AT)

def train():
  # TODO: Load if there is a model present

  # Create environment
  environment = gym.make("Breakout-v0")
  input_size = environment.observation_space.shape[0]
  output_size = environment.action_space.n

  # Create model
  state = tf.placeholder(tf.float32, [None, 84, 84, 4], name="state")

  with tf.name_scope("conv1"):
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(state, W_conv1, 4) + b_conv1)
  
  with tf.name_scope("conv2"):
    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 2) + b_conv2)
  
  with tf.name_scope("conv3"):
    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])
    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)

  with tf.name_scope("flatten"):
    flat_conv3 = tf.reshape(h_conv3, [-1])

  # hidden1, W1, b1 = nn_layer(state, input_size, HIDDEN_SIZE_1, "hidden1")
  # hidden2, W2, b2 = nn_layer(hidden1, HIDDEN_SIZE_1, HIDDEN_SIZE_2, "hidden2")
  # Q, W3, b3 = nn_layer(hidden2, HIDDEN_SIZE_2, output_size, "output_q", act=tf.identity)
  # weights = [W1, b1, W2, b2, W3, b3]

  s = environment.reset()
  s = tf.image.resize_images(s, [84, 84]).eval()

  

  targetQ = tf.placeholder(tf.float32, [None], name="targetQ")
  tf.summary.scalar("targetQ", targetQ)

  targetActionMask = tf.placeholder(tf.float32, [None, output_size], name="targetActionMask")
  maskedQ = tf.reduce_sum(tf.multiply(Q, targetActionMask), reduction_indices=[1], name="maskedQ")
  tf.summary.scalar("maskedQ", maskedQ)
  loss = tf.reduce_mean(tf.square(tf.subtract(maskedQ, targetQ)), name="loss")

  optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
  train_op = optimizer.minimize(loss)

  # Merge summaries
  merged = tf.summary.merge_all()
  tf.global_variables_initializer().run()

  # Write summary
  summary_writer = tf.summary.FileWriter(LOG_DIR, session.graph)

  epsilon = EPSILON_MAX
  replay_memory = []

  model_saver = tf.train.Saver()

  total_steps = 0

  target_weights = session.run(weights)
  
  # Run episodes
  for episode in range(NUMBER_OF_EPISODES):
    # Reset environment
    state = environment.reset()
    # Take steps
    for step in range(MAX_STEPS):
      # Pick the next action and execute it
      action = None
      epsilon = update_epsilon(total_steps)
      if random.random() < epsilon:
        action = environment.action_space.sample()
      else:
        q_values = session.run(Q, feed_dict={input_state: [state]})
        action = q_values.argmax()

      # Observation
      observation, reward, done, _ = environment.step(action)
      total_steps += 1

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
        feed_dict.update(zip(weights, target_weights))
        q_values = session.run(Q, feed_dict=feed_dict)
        max_q_values = q_values.max(axis=1)

#         # Sample a random minibatch and fetch max Q at s'
#         if len(self.replay_memory) >= 10000:
#           minibatch = random.sample(self.replay_memory, self.MINIBATCH_SIZE)
#           next_states = [m[3] for m in minibatch]
#           # TODO: Optimize to skip terminal states
#           feed_dict = {self.x: next_states}
#           feed_dict.update(zip(self.weights, target_weights))
#           q_values = self.session.run(self.Q, feed_dict=feed_dict)
#           max_q_values = q_values.max(axis=1)

#           # Compute target Q values
#           target_q = np.zeros(self.MINIBATCH_SIZE)
#           target_action_mask = np.zeros((self.MINIBATCH_SIZE, self.output_size), dtype=int)
#           for i in range(self.MINIBATCH_SIZE):
#             _, action, reward, _, terminal = minibatch[i]
#             target_q[i] = reward
#             if not terminal:
#               target_q[i] += self.DISCOUNT_FACTOR * max_q_values[i]
#             target_action_mask[i][action] = 1

#           # Gradient descent
#           states = [m[0] for m in minibatch]
#           feed_dict = {
#             self.x: states, 
#             self.targetQ: target_q,
#             self.targetActionMask: target_action_mask,
#           }
#           _, summary = self.session.run([self.train_op, self.summary], 
#               feed_dict=feed_dict)

#           # Write summary for TensorBoard
#           if total_steps % 1000 == 0:
#             self.summary_writer.add_summary(summary, total_steps)

#           # Update target network
#           if step % self.TARGET_UPDATE_FREQ == 0:
#             target_weights = self.session.run(self.weights)

#         total_steps += 1
#         steps += 1
#         if done:
#           break

#       step_counts.append(steps) 
#       mean_steps = np.mean(step_counts[-100:])
#       print("Training episode = {}, Total steps = {}, Last-100 mean steps = {}, epsilon: {}"
#         .format(episode, total_steps, mean_steps, self.random_action_prob))

#       if total_steps > self.END_AT_TOTAL_STEPS:
#         break


#   

#   def play(self):
#     state = self.env.reset()
#     done = False
#     steps = 0
#     while not done and steps < 1000:
#       self.env.render()
#       q_values = self.session.run(self.Q, feed_dict={self.x: [state]})
#       action = q_values.argmax()
#       state, _, done, _ = self.env.step(action)
#       steps += 1
#     return steps

if __name__ == "__main__":
  train()
  # dqn = DQN('CartPole-v0')
  # dqn.init_network()

  # dqn.train()
  # save_path = dqn.saver.save(dqn.session, "/tmp/dqnmodel.ckpt")
  # print("Model saved in file: %s" % save_path)

  # res = []
  # for i in range(100):
  #   steps = dqn.play()
  #   print("Test steps = ", steps)
  #   res.append(steps)
  # print("Mean steps = ", sum(res) / len(res))