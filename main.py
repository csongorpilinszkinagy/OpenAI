import gym
import random

import numpy as np
import tensorflow as tf

import time

class DQN:
  REPLAY_MEMORY_SIZE = 1000000
  RANDOM_ACTION_DECAY = 0.999995
  EPSILON_MIN = 0.1
  EPSILON_MAX = 1.0
  HIDDEN1_SIZE = 32
  HIDDEN2_SIZE = 32
  NUM_EPISODES = 30000
  MAX_STEPS = 1000
  LEARNING_RATE = 0.00025
  MINIBATCH_SIZE = 32
  DISCOUNT_FACTOR = 0.99
  TARGET_UPDATE_FREQ = 10000
  REG_FACTOR = 0.001
  LOG_DIR = '/tmp/dqn'
  DROPOUT = 0.9
  START_UPDATE_AT = 10000
  END_UPDATE_AT = 50000
  END_AT_TOTAL_STEPS = 100000

  random_action_prob = EPSILON_MAX
  replay_memory = []

  def __init__(self, env):
    self.env = gym.make(env)
    assert len(self.env.observation_space.shape) == 1
    self.input_size = self.env.observation_space.shape[0]
    self.output_size = self.env.action_space.n

  def weight_variable(self, shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

  def bias_variable(self, shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

  def variable_summaries(self, var):
    with tf.name_scope("summaries"):
      mean = tf.reduce_mean(var)
      tf.summary.scalar("mean", mean)
      with tf.name_scope("stddev"):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
      tf.summary.scalar("stddev", stddev)
      tf.summary.scalar("min", tf.reduce_min(var))
      tf.summary.scalar("max", tf.reduce_max(var))
      tf.summary.histogram("histogram", var)

  def nn_layer(self, input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
      with tf.name_scope("weights"):
        weights = self.weight_variable([input_dim, output_dim])
        self.variable_summaries(weights)
      with tf.name_scope("biases"):
        biases = self.bias_variable([output_dim])
        self.variable_summaries(biases)
      with tf.name_scope("Wx_plus_b"):
        preactivate = tf.matmul(input_tensor, weights) + biases
        tf.summary.histogram("pre_activations", preactivate)
      activations = act(preactivate, name="activation")
      tf.summary.histogram("activations", activations)
      return activations


  def init_network(self):
    self.session = tf.Session()
    # Inference
    self.x = tf.placeholder(tf.float32, [None, self.input_size])
    hidden1 = self.nn_layer(self.x, self.input_size, self.HIDDEN1_SIZE, "hidden1")
    hidden2 = self.nn_layer(hidden1, self.HIDDEN1_SIZE, self.HIDDEN2_SIZE, "hidden2")
    self.Q = self.nn_layer(hidden2, self.HIDDEN2_SIZE, self.output_size, "output", act=tf.identity)
    self.weights = []

    # Loss
    self.targetQ = tf.placeholder(tf.float32, [None])
    self.targetActionMask = tf.placeholder(tf.float32, [None, self.output_size])
    # TODO: Optimize this
    q_values = tf.reduce_sum(tf.multiply(self.Q, self.targetActionMask),reduction_indices=[1])
    self.loss = tf.reduce_mean(tf.square(tf.subtract(q_values, self.targetQ)))

    # Training
    optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    self.train_op = optimizer.minimize(self.loss, global_step=global_step)

    self.saver = tf.train.Saver()

  def train(self, num_episodes=NUM_EPISODES):

    # Summary for TensorBoard
    tf.summary.scalar('loss', self.loss)
    tf.summary.histogram('Q_values', self.Q)
    self.summary = tf.summary.merge_all()
    self.summary_writer = tf.summary.FileWriter(self.LOG_DIR, self.session.graph)

    self.session.run(tf.global_variables_initializer())
    total_steps = 0
    step_counts = []

    target_weights = self.session.run(self.weights)
    for episode in range(num_episodes):
      state = self.env.reset()
      steps = 0

      for step in range(self.MAX_STEPS):
        # Pick the next action and execute it
        action = None
        if random.random() < self.random_action_prob:
          action = self.env.action_space.sample()
        else:
          q_values = self.session.run(self.Q, feed_dict={self.x: [state]})
          action = q_values.argmax()
        self.random_action_prob = self.update_epsilon(total_steps, self.START_UPDATE_AT, self.EPSILON_MAX, self.END_UPDATE_AT, self.EPSILON_MIN)
        obs, reward, done, _ = self.env.step(action)

        # Update replay memory
        self.replay_memory.append((state, action, reward, obs, done))
        if len(self.replay_memory) > self.REPLAY_MEMORY_SIZE:
          self.replay_memory.pop(0)
        state = obs

        # Sample a random minibatch and fetch max Q at s'
        if len(self.replay_memory) >= 10000:
          minibatch = random.sample(self.replay_memory, self.MINIBATCH_SIZE)
          next_states = [m[3] for m in minibatch]
          # TODO: Optimize to skip terminal states
          feed_dict = {self.x: next_states}
          feed_dict.update(zip(self.weights, target_weights))
          q_values = self.session.run(self.Q, feed_dict=feed_dict)
          max_q_values = q_values.max(axis=1)

          # Compute target Q values
          target_q = np.zeros(self.MINIBATCH_SIZE)
          target_action_mask = np.zeros((self.MINIBATCH_SIZE, self.output_size), dtype=int)
          for i in range(self.MINIBATCH_SIZE):
            _, action, reward, _, terminal = minibatch[i]
            target_q[i] = reward
            if not terminal:
              target_q[i] += self.DISCOUNT_FACTOR * max_q_values[i]
            target_action_mask[i][action] = 1

          # Gradient descent
          states = [m[0] for m in minibatch]
          feed_dict = {
            self.x: states, 
            self.targetQ: target_q,
            self.targetActionMask: target_action_mask,
          }
          _, summary = self.session.run([self.train_op, self.summary], 
              feed_dict=feed_dict)

          # Write summary for TensorBoard
          if total_steps % 1000 == 0:
            self.summary_writer.add_summary(summary, total_steps)

          # Update target network
          if step % self.TARGET_UPDATE_FREQ == 0:
            target_weights = self.session.run(self.weights)

        total_steps += 1
        steps += 1
        if done:
          break

      step_counts.append(steps) 
      mean_steps = np.mean(step_counts[-100:])
      print("Training episode = {}, Total steps = {}, Last-100 mean steps = {}, epsilon: {}"
        .format(episode, total_steps, mean_steps, self.random_action_prob))

      if total_steps > self.END_AT_TOTAL_STEPS:
        break


  def update_epsilon(self, total_steps, start_at, start_value, end_at, end_value):
    if total_steps < start_at:
      return start_value
    if total_steps > end_at:
      return end_value
    return start_value + (end_value - start_value) / (end_at - start_at) * (total_steps - start_at)

  def play(self):
    state = self.env.reset()
    done = False
    steps = 0
    while not done and steps < 1000:
      self.env.render()
      q_values = self.session.run(self.Q, feed_dict={self.x: [state]})
      action = q_values.argmax()
      state, _, done, _ = self.env.step(action)
      steps += 1
    return steps

if __name__ == "__main__":
  dqn = DQN('CartPole-v0')
  dqn.init_network()

  dqn.train()
  save_path = dqn.saver.save(dqn.session, "/tmp/dqnmodel.ckpt")
  print("Model saved in file: %s" % save_path)

  res = []
  for i in range(100):
    steps = dqn.play()
    print("Test steps = ", steps)
    res.append(steps)
  print("Mean steps = ", sum(res) / len(res))