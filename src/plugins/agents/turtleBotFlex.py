"""Factory for 'turtleBotFlex' agent (auto-generated)."""

from plugins.agents import register_agent
from robots.turtleBotFlex.agent import TurtlebotflexAgent

def make_agent(sim, params_env, paths, file_id, verbose, comms_port=49054):
    """Return an instance of the robot-specific Agent.

    Args:
        sim: Coppelia API object.
        params_env (dict): Environment parameters.
        paths (dict): Project paths.
        file_id (str): Experiment/session identifier.
        verbose (int): Verbosity level.
        comms_port (int): RL comms port.

    Returns:
        TurtlebotflexAgent: Configured agent instance.
    """
    return TurtlebotflexAgent(sim, params_env, paths, file_id, verbose, comms_port)

register_agent("turtleBotFlex", make_agent)
