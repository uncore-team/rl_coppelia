"""Environment plugin registry (RL side).

Each module inside this package (plugins/envs/<robot>.py) should call
`register_env(robot_name, make_env)` on import.

Example:
    from plugins.envs import register_env
    from robots.burgerBot.envs import BurgerBotEnv
    from stable_baselines3.common.env_util import make_vec_env

    def make_env(manager):
        return make_vec_env(
            BurgerBotEnv,
            n_envs=1,
            monitor_dir=manager.log_monitor,
            env_kwargs={
                "params_env": manager.params_env,
                "comms_port": manager.free_comms_port,
            },
        )

    register_env("burgerBot", make_env)
"""

from typing import Callable, Dict, Any

_ENV_FACTORIES: Dict[str, Callable[..., Any]] = {}


def register_env(name: str, factory: Callable[..., Any]) -> None:
    """Register a new environment factory by robot name."""
    _ENV_FACTORIES[name] = factory


def get_env_factory(name: str) -> Callable[..., Any] | None:
    """Return the factory for a robot, or None if not found."""
    return _ENV_FACTORIES.get(name)