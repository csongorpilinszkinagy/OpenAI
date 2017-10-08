import gym
import random

import numpy as np
import tensorflow as tf

from PIL import Image

REPLAY_MEMORY_SIZE = 1000000
EPSILON_MIN = 0.1
EPSILON_MAX = 1.0
NUMBER_OF_EPISODES = 10
END_AT_TOTAL_STEPS = 2000000
START_UPDATE_AT = 10000
END_UPDATE_AT = 100000
MAX_STEPS = 10000
LEARNING_RATE = 0.001
MINIBATCH_SIZE = 32
DISCOUNT_FACTOR = 0.99
TARGET_UPDATE_FREQ = 10000
LOG_DIR = "/Users/pilinszki-nagycsongor/Developer/OpenAI/save/"

MINIMUM_SAMPLE_SIZE = 10000

def weight_variable(shape):
  return tf.Variable(tf.truncated_normal(shape))

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
    activations = tf.maximum(preactivate, 0.1 * preactivate)
    tf.summary.histogram("activations", activations)
    return activations, weights, biases

def update_epsilon(total_steps):
  if total_steps < START_UPDATE_AT:
    return EPSILON_MAX
  if total_steps > END_UPDATE_AT:
    return EPSILON_MIN
  return EPSILON_MAX + (EPSILON_MIN - EPSILON_MAX) / (END_UPDATE_AT - START_UPDATE_AT) * (total_steps - START_UPDATE_AT)

def format_image(image):
  gray_image = tf.image.rgb_to_grayscale(image)
  resized_image = tf.image.resize_images(gray_image, [84, 84], method=tf.image.ResizeMethod.AREA)
  image_2d = np.squeeze(resized_image.eval(), axis=2)
  return image_2d

def train():
  # Create session
  session = tf.InteractiveSession()

  # TODO: Load if there is a model present

  # Create environment
  environment = gym.make("Breakout-v0")
  input_size = environment.observation_space.shape[0]
  output_size = environment.action_space.n

  # TODO: parametrize sizes
  with tf.name_scope("input"):
    input_state = tf.placeholder(tf.float32, [None, 84, 84, 4], name="input_state")
    tf.summary.histogram("input_state", input_state)
    targetQ = tf.placeholder(tf.float32, [None], name="targetQ")
    targetActionMask = tf.placeholder(tf.float32, [None, output_size], name="targetActionMask")

  with tf.name_scope("conv1"):
    W_conv1 = tf.get_variable("W_conv1", shape=[8, 8, 4, 32], initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False))
    b_conv1 = bias_variable([32])
    m_conv1 = conv2d(input_state, W_conv1, 4) + b_conv1
    h_conv1 = tf.maximum(m_conv1, 0.1 * m_conv1)

    tf.summary.histogram("W_conv1", W_conv1)
    tf.summary.histogram("b_conv1", b_conv1)
    tf.summary.histogram("m_conv1", m_conv1)
    tf.summary.histogram("h_conv1", h_conv1)
  
  with tf.name_scope("conv2"):
    W_conv2 = tf.get_variable("W_conv2", shape=[4, 4, 32, 64], initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False))
    b_conv2 = bias_variable([64])
    m_conv2 = conv2d(h_conv1, W_conv2, 2) + b_conv2
    h_conv2 = tf.maximum(m_conv2, 0.1 * m_conv2)

    tf.summary.histogram("W_conv2", W_conv2)
    tf.summary.histogram("b_conv2", b_conv2)
    tf.summary.histogram("m_conv2", m_conv2)
    tf.summary.histogram("h_conv2", h_conv2)
  
  with tf.name_scope("conv3"):
    W_conv3 = tf.get_variable("W_conv3", shape=[3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False))
    b_conv3 = bias_variable([64])
    m_conv3 = conv2d(h_conv2, W_conv3, 1) + b_conv3
    h_conv3 = tf.maximum(m_conv3, 0.1 * m_conv3)

    tf.summary.histogram("W_conv3", W_conv3)
    tf.summary.histogram("b_conv3", b_conv3)
    tf.summary.histogram("m_conv3", m_conv3)
    tf.summary.histogram("h_conv3", h_conv3)

  with tf.name_scope("flatten"):
    flat_conv3 = tf.reshape(h_conv3, [-1, 7744])

  hidden, W1, b1 = nn_layer(flat_conv3, 7744, 512, "hidden")
  Q, W2, b2 = nn_layer(hidden, 512, output_size, "output_q", act=tf.identity)
  weights = [W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W1, b1, W2, b2]
  
  with tf.name_scope("Q_learning"):
    maskedQ = tf.reduce_sum(tf.multiply(Q, targetActionMask), reduction_indices=[1], name="maskedQ")
    loss = tf.reduce_mean(tf.square(tf.subtract(maskedQ, targetQ)), name="loss")

    tf.summary.scalar("loss", loss)

  with tf.name_scope("train"):
    train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

  # Merge summaries
  merged_summary = tf.summary.merge_all()
  tf.global_variables_initializer().run()

  # Summary writer and model saver
  summary_writer = tf.summary.FileWriter(LOG_DIR, session.graph)
  model_saver = tf.train.Saver()

  # TODO: if there is a model load it
  # model_saver.restore(session, LOG_DIR + "model.ckpt")

  epsilon = EPSILON_MAX
  replay_memory = []

  total_steps = 0
  fixed_weights = session.run(weights)
  
  # Run episodes
  for episode in range(NUMBER_OF_EPISODES):
    # Reset environment
    image = environment.reset()
    small_image =  format_image(image)
    normalized_image = (small_image - 128.0) / 128.0
    image_array = [normalized_image, normalized_image, normalized_image, normalized_image]
    state = np.stack(image_array, axis=2)
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

      image, reward, done, _ = environment.step(action)
      small_image = format_image(image)
      normalized_image = (small_image - 128.0) / 128.0

      image_array.pop(0)
      image_array.append(normalized_image)
      observation = np.stack(image_array, axis=2)

      total_steps += 1
      print total_steps

      # Update replay memory
      if total_steps < 50:
        replay_memory.append((state, action, reward, observation, done))
      if total_steps > REPLAY_MEMORY_SIZE:
        replay_memory.pop(0)
      
      state = observation

      # Sample a random minibatch
      if total_steps > MINIBATCH_SIZE:
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

        # Gradient descent
        states = [m[0] for m in minibatch]
        feed_dict = {
          input_state: states, 
          targetQ: target_q,
          targetActionMask: target_action_mask,
        }
        _, summary = session.run([train_op, merged_summary], 
            feed_dict=feed_dict)

        # Write summary for TensorBoard
        if total_steps % 1 == 0:
          summary_writer.add_summary(summary, total_steps)

        # Update target network
        if step % 1000 == 0:
          fixed_weights = session.run(weights)

      if done:
        break

#       step_counts.append(steps) 
#       mean_steps = np.mean(step_counts[-100:])
#       print("Training episode = {}, Total steps = {}, Last-100 mean steps = {}, epsilon: {}"
#         .format(episode, total_steps, mean_steps, self.random_action_prob))

#       if total_steps > self.END_AT_TOTAL_STEPS:
#         break


#   

def play_random():
  env = gym.make("Breakout-v0")
  state = env.reset()
  done = False
  steps = 0
  for i in range(1000):
    env.render()
    _, _, done, _ = env.step(env.action_space.sample())
    if done:
      break
    time.sleep(1)
    
  return steps

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