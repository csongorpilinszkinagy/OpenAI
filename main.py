import gym

# Define constants
ENV_NAME = "CartPole-v0"
LR = 1e-3
MAX_STEPS = 500
SCORE_REQUIREMENT = 50
INITIAL_GAMES = 10000
NUMBER_OF_EPISODES = 5

# Initializing the environment
env = gym.make(ENV_NAME)
env.reset()

# Random agent
def randomAgent():
    for episode in range(NUMBER_OF_EPISODES):
        env.reset()
        for t in range(MAX_STEPS):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                break

randomAgent()