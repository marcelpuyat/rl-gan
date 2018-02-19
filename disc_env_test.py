import gym
import drawing_env
import numpy as np

env = gym.make('DrawEnv-v0')

for i in xrange(20000):
	if i > 8:
		i = 8
	env.step((2, i))