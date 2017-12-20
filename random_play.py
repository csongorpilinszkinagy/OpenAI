import gym
import numpy as np

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
  print("std: %s" % np.std(rewards_array))
  print("Maximum: %s" % max(rewards_array))

  return

if __name__ == "__main__":
    play_random()