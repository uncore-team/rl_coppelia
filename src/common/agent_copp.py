import logging
import os
import shutil
import sys

project_folder = shutil.which("rl_coppelia")
common_paths = [
    os.path.expanduser("~/Documents"), ("~/Downloads"), ("~/devel")]

project_path = None

for path in common_paths:
    for root, dirs, files in os.walk(path):
        if "rl_coppelia" in dirs: 
            project_path = os.path.join(root, "rl_coppelia") 
            break  
    if project_path:
        break
sys.path.append(os.path.abspath(os.path.join(project_path,"src")))

from common import utils
from common.coppelia_agents import BurgerBotAgent, TurtleBotAgent


def sysCall_init():
    """
    Initialize the simulation.
    """
    
    
def sysCall_thread():
    """
    Main loop to continuously handle instructions and actions.
    """    
    # Define comm_side, robot name and base path
    comm_side = "agent"
    robot_name = "turtleBot"
    model_name = None
    comms_port = 49054
    base_path = ""
    params_env = {}
    
    # Instantiate the simulation object
    sim = require('sim')    # type: ignore
    
    # Generate tf and log paths
    paths = utils.get_robot_paths(base_path, robot_name, just_agent_logs = True)
    
    # Get the id of the upcoming script logs
    file_id = utils.get_file_index (model_name, paths["tf_logs"], robot_name)
    utils.logging_config(paths["script_logs"],comm_side, robot_name, file_id, log_level=logging.INFO, verbose=2)
         
    logging.info(" ----- START EXPERIMENT ----- ")
    
    # Instantiate agent object
    if robot_name == "turtleBot":
        agent = TurtleBotAgent(sim, params_env, comms_port=comms_port)
    elif robot_name == "burgerBot":
        agent = BurgerBotAgent(sim, params_env, comms_port=comms_port)
        agent.robot_baselink = agent.robot

    # Loop for processing instructions from RL continuously until the agent receives a FINISH command.
    while not agent.finish_rec:
        logging.debug("Waiting for instructions...")
        action = agent.agent_step()
        # If an action is received, execute it
        if action is not None:
            # With this check we avoid calling cmd_vel script repeteadly for the same action
            # if agent.execute_cmd_vel:
            if len(action)>0:
                agent.sim.callScriptFunction('cmd_vel', agent.handle_cmd_vel_script, action["linear"], action["angular"], agent.colorID)
            if agent._waitingforrlcommands:
                agent.colorID +=1
                # agent.execute_cmd_vel = False
        # FINISH command --> Break the loop
        if agent.finish_rec:
            break

    # Stop the robot
    sim.callScriptFunction('cmd_vel',agent.handle_cmd_vel_script,0,0)

    logging.info(" ----- END OF EXPERIMENT ----- ")