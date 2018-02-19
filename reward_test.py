import gym
import drawing_env
import numpy as np

env = gym.make('DrawEnv-v0')

state, reward, done, _ = env.step((1, 0))
print "Reward: " + str(reward)
assert reward > 0