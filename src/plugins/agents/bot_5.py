"""Factory for 'bot_5' agent (auto-generated)."""

from plugins.agents import register_agent
from robots.bot_5.agent import Bot5Agent

def make_agent(sim, params_robot, params_env, paths, file_id, verbose, comms_port=49054):
    """Return an instance of the robot-specific Agent.

    Args:
        sim: Coppelia API object.
        params_robot (dict): Robot-specific parameters (e.g., robot_radius).
        params_env (dict): Environment parameters.
        paths (dict): Project paths.
        file_id (str): Experiment/session identifier.
        verbose (int): Verbosity level.
        comms_port (int): RL comms port.

    Returns:
        Bot5Agent: Configured agent instance.
    """
    return Bot5Agent(sim, params_robot, params_env, paths, file_id, verbose, comms_port)

register_agent("bot_5", make_agent)
