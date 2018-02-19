from gym.envs.registration import register

register(
    id='DrawEnv-v0',
    entry_point='drawing_env.envs:DrawEnv',
)