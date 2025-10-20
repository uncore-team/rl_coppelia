"""Plugin to register 'robot_turtle_v21' robot environment.

Loaded on the RL side. It registers a VecEnv factory on import.
"""

from __future__ import annotations
from common.rl_coppelia_manager import RLCoppeliaManager
from plugins.envs import register_env
from stable_baselines3.common.env_util import make_vec_env
from robots.robot_turtle_v21.envs import RobotTurtleV21Env

def make_env(manager: RLCoppeliaManager):
    """Create a VecEnv instance for 'robot_turtle_v21'.

    Args:
        manager: The current RLCoppeliaManager instance.

    Returns:
        A vectorized environment (VecEnv) suitable for training/testing.
    """
    return make_vec_env(
        RobotTurtleV21Env,
        n_envs=1,
        monitor_dir=manager.log_monitor,
        env_kwargs={
            "params_env": manager.params_env,
            "comms_port": manager.free_comms_port,
        },
    )

# Register on module import
register_env("robot_turtle_v21", make_env)
