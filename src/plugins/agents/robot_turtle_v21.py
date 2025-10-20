"""Factory for 'robot_turtle_v21' agent (auto-generated)."""

from plugins.agents import register_agent
from robots.robot_turtle_v21.agent import RobotTurtleV21Agent

def make_agent(sim, params_env, paths, file_id, verbose, comms_port=49054):
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
        RobotTurtleV21Agent: Configured agent instance.
    """
    return RobotTurtleV21Agent(sim, params_env, paths, file_id, verbose, comms_port)

register_agent("robot_turtle_v21", make_agent)
