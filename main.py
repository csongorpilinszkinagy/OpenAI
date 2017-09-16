import gym
import tensorflow as tf

LOGDIR = "/tmp/dqn/"

# Define constants
ENV_NAME = "CartPole-v0"
MAX_STEPS = 500
NUMBER_OF_EPISODES = 10

class agent:
    # Define constants
    REPLAY_MEMORY_SIZE = 10000
    EPSILON_MIN = 0.01
    EPSILON_MAX = 1
    LEARNING_RATE_MIN = 0.001
    LEARNING_RATE_MAX = 0.01
    HIDDEN_SIZE = 20
    NUMBER_OF_EPISODES = 1000
    MAX_STEPS = 500
    MINIBATCH_SIZE = 30
    DISCOUNT_FACTOR = 0.99
    TARGET_UPDATE_FREQ= 500
    STDDEV = 0.01

    def __init__(self, environment):
        self.environment = environment
        self.input_size = self.environment.observation_space.shape[0]
        self.output_size = self.environment.action_space.n
        self.replay_memory = []

    def init_network(self):
        
        tf.reset_default_graph()
        self.session = tf.Session()

        #Setup placeholders
        self.input = tf.placeholder(tf.float32, shape=[None, self.input_size], name="input")
        self.targetQ = tf.placeholder(tf.float32, [None], name="target_q")
        self.targetActionMask = tf.placeholder(tf.float32, [None, self.output_size], name="target_action_mask")

        hidden_layer = fc_relu_layer(self.input, self.input_size, self.HIDDEN_SIZE, name="hidden_layer")
        self.q = fc_layer(hidden_layer, self.HIDDEN_SIZE, self.output_size, name="q")
        
        # TODO: Optimize this
        with tf.name_scope("loss"):
            q_values = tf.reduce_sum(tf.matmul(self.q, self.targetActionMask), reduction_indices=[1])
            self.loss = tf.reduce_mean(tf.square(tf.subtract(q_values, self.targetQ)))
            tf.summary.scalar("loss", loss)

        with tf.name_scope("train"):
            self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        self.summary = tf.merge_all_summaries()
        self.summary_writer = tf.summary.FileWriter(LOGDIR)
        self.summary_writer.add_graph(sess.graph)
        self.summary_writer.add_summary(self.summary)

    # def play():
        

    #     # Initializing the environment
    #     env = gym.make(ENV_NAME)
    #     env.reset()

    #     for episode in range(NUMBER_OF_EPISODES):
    #         env.reset()
    #         for t in range(MAX_STEPS):
    #             env.render()
    #             action = env.action_space.sample()
    #             observation, reward, done, info = env.step(action)
    #             if done:
    #                 break

    def fc_layer(input, size_in, size_out, name="fc_layer"):
        with tf.name_scope(name):
            w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="w")
            b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="b")
            act = tf.matmul(input, w) + b
            tf.summary.histogram("weights", w)
            tf.summary.histogram("biases", b)
            tf.summary.histogram("activations", act)
            return act

    def fc_relu_layer(input, size_in, size_out, name="fc_relu_layer"):
        with tf.name_scope(name):
            relu = tf.nn.relu(self.fc_layer(input, size_in, size_out))
            tf.summary.histogram("relu", relu)
            return relu

    def random_steps(self):
        state = self.environment.reset()
        print self.output_size
        for i in range(100):
            action = self.environment.action_space.sample()
            observation, reward, done, info = self.environment.step(action)
            self.replay_memory.append((state, action, reward, observation, done))
            state = observation

            if done:
                break

        print self.replay_memory

def main():
    environment = gym.make("CartPole-v0")
    cartpole_agent = agent(environment)
    cartpole_agent.random_steps()


if __name__ == "__main__":
    main()