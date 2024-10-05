from gym.envs.registration import (
    registry,
    register,
    make,
    spec,
    load_env_plugins as _load_env_plugins,
)

_load_env_plugins()


register(id='lorenz_le-v0',entry_point='Custom_envs.envs:Lorenzle', max_episode_steps=200,)

register(id='pendulum_le-v0',entry_point='Custom_envs.envs:pendulum_le', max_episode_steps=200,)

register(id='cartpole_le-v0',entry_point='Custom_envs.envs:cartpole_le', max_episode_steps=200,)

register(id='double_pendulum_le-v0',entry_point='Custom_envs.envs:double_pendulum_le', max_episode_steps=200,)
