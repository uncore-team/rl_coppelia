"""Plugin to register 'bot_5' robot environment.

Loaded on the RL side. It registers a VecEnv factory on import.
"""

from plugins.envs import register_env
from stable_baselines3.common.env_util import make_vec_env
from robots.bot_5.envs import Bot5Env

def make_env(manager: RLCoppeliaManager):
    """Create a VecEnv instance for 'bot_5'.

    Args:
        manager: The current RLCoppeliaManager instance.

    Returns:
        A vectorized environment (VecEnv) suitable for training/testing.
    """
    return make_vec_env(
        Bot5Env,
        n_envs=1,
        monitor_dir=manager.log_monitor,
        env_kwargs={
            "params_env": manager.params_env,
            "comms_port": manager.free_comms_port,
        },
    )

# Register on module import
register_env("bot_5", make_env)
