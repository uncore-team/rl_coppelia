"""Agent plugin registry (Coppelia side).

Each module inside this package (plugins/agents/<robot>.py) should call
`register_agent(robot_name, make_agent)` on import.

Example usage in Coppelia:
    from plugins.agents import get_agent_factory
    factory = get_agent_factory("turtleBot")
    if factory:
        agent = factory(sim, params_env, paths, file_id, verbose, comms_port)
"""

from typing import Callable, Dict, Any

_AGENT_FACTORIES: Dict[str, Callable[..., Any]] = {}


def register_agent(name: str, factory: Callable[..., Any]) -> None:
    """Register a new agent factory by robot name."""
    _AGENT_FACTORIES[name] = factory


def get_agent_factory(name: str) -> Callable[..., Any] | None:
    """Return the factory for a given robot, or None if not found."""
    return _AGENT_FACTORIES.get(name)
