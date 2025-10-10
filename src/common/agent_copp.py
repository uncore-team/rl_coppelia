"""
This script serves as the simulation-side handler for CoppeliaSim integration in the RL-Coppelia framework.
It initializes the agent based on robot type, handles the interaction between the agent and the environment, 
saves scene configurations and robot trajectories, and manages control logic through simulation lifecycle callbacks.

Usage:
    This script is designed to be used within CoppeliaSim's threaded environment and expects to be called via:
        - sysCall_init(): Initialization logic.
        - sysCall_thread(): Agent setup before simulation steps begin.
        - sysCall_sensing(): Executes each simulation step, processes actions, and stores data.

Features:
    - Dynamically finds the RL-Coppelia source directory and loads common utilities and agent classes.
    - Automatically creates necessary directories for logs and scene saving.
    - Supports both TurtleBot and BurgerBot agents.
    - Communicates with an external RL training process using a communication port.
    - Captures robot trajectories and can save scene configurations.
"""

import logging
import os
import shutil
import sys

# Locate rl_coppelia installation and append the source folder to sys.path
project_folder = shutil.which("rl_coppelia")
common_paths = [
        os.path.expanduser("~/Documents"),  # Search in Documents folder
        os.path.expanduser("~/Downloads"),  # Search in Downloads folder
        os.path.expanduser("~/devel"),
        "/opt", "/usr/local", "/home"     # Common system directories
    ]

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

# Global variables for the simulation context
sim = None
agent = None
verbose = 1
sim_initialized = False
agent_created = False


def sysCall_init():
    """
    Called at the beginning of the simulation to configure logging and path setup. It alseo receives variable data from the RL side.
    """
    global sim, agent, verbose, sim_initialized, robot_name, model_ids, params_env, comms_port, paths, file_id, scene_to_load_folder, save_scene, save_traj, action_times, model_name, test_scene_mode
    sim = require('sim')    # type: ignore

    # Variables to get from agent_copp.py script
    comm_side = "agent"
    robot_name = "turtleBot"
    model_name = None
    model_ids = None
    comms_port = 49054
    base_path = ""
    params_env = {}
    verbose = 1
    scene_to_load_folder = ""
    save_scene = None
    save_traj = None
    action_times = None
    test_scene_mode = ""

    # Send verbose value to RObot_Script
    sim.setInt32Signal('verboseValue', verbose)

    # Generate needed routes for logs and tf
    paths = utils.get_robot_paths(base_path, robot_name, agent_logs=True)
    file_id = utils.get_file_index(model_name, paths["tf_logs"], robot_name)
    utils.logging_config(paths["script_logs"], comm_side, robot_name, file_id, log_level=logging.INFO, verbose=verbose)

    logging.info(" ----- START EXPERIMENT ----- ")
    sim_initialized = True


def sysCall_thread():
    """
    Called once after simulation starts to create the agent and configure paths.
    """
    global sim, agent, robot_name, params_env, comms_port, sim_initialized, model_ids, paths, file_id, scene_to_load_folder, save_scene, save_traj, agent_created, action_times, model_name, verbose, test_scene_mode

    if sim_initialized:
        logging.info("inside thread sim_initialized")
        # Create agent
        if robot_name == "turtleBot":
            agent = TurtleBotAgent(sim, params_env, paths, file_id, verbose, comms_port=comms_port)
        elif robot_name == "burgerBot":
            agent = BurgerBotAgent(sim, params_env, paths, file_id, verbose, comms_port=comms_port)
            agent.robot_baselink = agent.robot
        else:   # by default it will use the BurgerBot configuration
            agent = BurgerBotAgent(sim, params_env, paths, file_id, verbose, comms_port=comms_port)
        logging.info("Agent initialized")

        # Configure scene and trajectory behavior
        agent.scene_to_load_folder = scene_to_load_folder
        agent.save_scene = save_scene
        agent.model_ids = model_ids
        agent.action_times = action_times
        agent.test_scene_mode = test_scene_mode

        # Set the folder where the trajectories will be saved (inside testing_metrics folder)
        if model_name is None:
            agent.save_traj_csv_folder = os.path.join(
            agent.scene_configs_path,
            agent.scene_to_load_folder,
            "trajs"
        )
        else:
            agent.save_traj_csv_folder = os.path.join(
                agent.save_trajs_path,
                f"{model_name}_testing",
                "trajs"
            )

        if agent.save_scene:
            os.makedirs(agent.save_scene_csv_folder, exist_ok=True)
            logging.info(f"Scene configurations will be saved in: {agent.save_scene_csv_folder}.")

        # Automatically force save_traj=True if loading from a predefined scene
        if agent.scene_to_load_folder =="" or agent.scene_to_load_folder is None:
            agent.save_traj = save_traj
            if agent.save_traj:
                os.makedirs(agent.save_traj_csv_folder, exist_ok=True)
                logging.info(f"Trajectories will be saved in: {agent.save_traj_csv_folder}.")
        else:
            agent.save_traj = True
            os.makedirs(agent.save_traj_csv_folder, exist_ok=True)
            logging.info(f"Scene configuration inside {agent.scene_to_load_folder} will be loaded and trajectory will be saved inside it.")

        agent_created = True


def sysCall_sensing():
    """
    Called at each simulation step. Executes the agent's action and logs trajectories.
    """
    global sim, agent, verbose, agent_created, robot_name
    
    if agent and not agent.finish_rec and agent_created:
        # Loop for processing instructions from RL continuously until the agent receives a FINISH command.
        action = agent.agent_step(robot_name)

        # If an action is received, execute it
        if action is not None:
            # With this check we avoid calling cmd_vel script repeteadly for the same action
            if agent.execute_cmd_vel:
                if len(action)>0:
                    logging.info("Execute action")
                    agent.sim.callScriptFunction('cmd_vel', agent.handle_robot_scripts, action["linear"], action["angular"])
            if verbose == 3:
                if len(action)>0:
                    agent.sim.callScriptFunction('draw_path', agent.handle_robot_scripts, action["linear"], action["angular"], agent.colorID)
                    if agent._waitingforrlcommands:
                        agent.colorID +=1
            agent.execute_cmd_vel = False
        
        # Save current robot position for later saving it csv file
        if agent.episode_idx >=1 and agent.save_traj:
            position = agent.sim.getObjectPosition(agent.robot_baselink, -1)
            agent.trajectory.append({"x": position[0], "y": position[1]})
            logging.debug(f"x pos: {position[0]}; y pos: {position[1]}")

    # FINISH command --> Finish the experiment
    if agent and agent.finish_rec:
        # Stop the robot
        logging.info("Reset speed to 0")
        sim.callScriptFunction('cmd_vel',agent.handle_robot_scripts,0,0)
        sim.callScriptFunction('draw_path', agent.handle_robot_scripts, 0,0, agent.colorID)

        logging.info(" ----- END OF EXPERIMENT ----- ")
        agent.finish_rec = False

    # Time tracking setup
    if agent and not agent.training_started:
        agent.initial_realTime = sim.getSystemTime()
        agent.initial_simTime = sim.getSimulationTime()
    
    elif agent and agent.training_started:
        simTime = sim.getSimulationTime() - agent.initial_simTime
        logging.debug("SIM Time:", simTime)
        realTime = sim.getSystemTime() - agent.initial_realTime
        logging.debug("REAL Time:", realTime)

    