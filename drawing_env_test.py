import gym
import drawing_env
import numpy as np

env = gym.make('DrawEnv-v0')

# We test that the state updates as we expect, the reward is as we expect,
# and that the `done` variable is only true once all values are non-zero.
env.render()
state, reward, done, _ = env.step((1, 0))
assert np.array_equal(state, np.array([1, 0, 0, 0, 0, 0, 0, 0, 0]))
assert reward == 0
assert done == False
env.render()
state, reward, done, _ = env.step((1, 1))
assert np.array_equal(state, np.array([1, 1, 0, 0, 0, 0, 0, 0, 0]))
assert reward == 0
assert done == False
env.render()
state, reward, done, _ = env.step((1, 2))
assert np.array_equal(state, np.array([1, 1, 1, 0, 0, 0, 0, 0, 0]))
assert reward == 0
assert done == False
env.render()
state, reward, done, _ = env.step((1, 3))
assert np.array_equal(state, np.array([1, 1, 1, 1, 0, 0, 0, 0, 0]))
assert reward == 0
assert done == False
env.render()
state, reward, done, _ = env.step((1, 4))
assert np.array_equal(state, np.array([1, 1, 1, 1, 1, 0, 0, 0, 0]))
assert reward == 0
assert done == False
env.render()
state, reward, done, _ = env.step((1, 5))
assert np.array_equal(state, np.array([1, 1, 1, 1, 1, 1, 0, 0, 0]))
assert reward == 0
assert done == False
env.render()
state, reward, done, _ = env.step((1, 6))
assert np.array_equal(state, np.array([1, 1, 1, 1, 1, 1, 1, 0, 0]))
assert reward == 0
assert done == False
env.render()
state, reward, done, _ = env.step((1, 7))
assert np.array_equal(state, np.array([1, 1, 1, 1, 1, 1, 1, 1, 0]))
assert reward == 0
assert done == False
env.render()
state, reward, done, _ = env.step((1, 8))
assert np.array_equal(state, np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]))
assert reward == 0
assert done == True
env.render()