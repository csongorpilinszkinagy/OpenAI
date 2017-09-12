import gym

# Random agent
class RandomAgent:
    # Define constants
    ENV_NAME = "CartPole-v0"
    MAX_STEPS = 500
    NUMBER_OF_EPISODES = 10

    def play():
        

        # Initializing the environment
        env = gym.make(ENV_NAME)
        env.reset()

        for episode in range(NUMBER_OF_EPISODES):
            env.reset()
            for t in range(MAX_STEPS):
                env.render()
                action = env.action_space.sample()
                observation, reward, done, info = env.step(action)
                if done:
                    break

# Deep Q Network
class DQN:
    # Define constants
    REPLAY_MEMORY_SIZE = 1000
    EPSILON_MIN = 0.01
    EPSILON_MAX = 1
    LEARNING_RATE_MIN = 0.001
    LEARNING_RATE_MAX = 0.01
    HIDDEN_SIZE = 20
    NUMBER_OF_EPISODES = 1000
    MAX_STEPS = 500
    MINIBATCH_SIZE = 30
    DISCOUNT_FACTOR = 0.99

    def __init__(self, env):
        self.env = gym.make(env)
        assert len(self.env.observation_space.shape) == 1
        self.input_size = self.env.observation_space.shape[0]
        self.output_size = self.env.action_space.n
        self.replay_memory = []

if __name__ == '__main__':
    dqn = DQN('CartPole-v0')
    print "Run succesful!"