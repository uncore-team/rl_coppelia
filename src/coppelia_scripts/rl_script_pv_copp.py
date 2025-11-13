# Agent_Script: minimal FSM for the Path-Probe experiment
# -------------------------------------------------------
# Responsibilities:
#   - Create the Agent (TurtleBot/BurgerBot or plugin).
#   - Open the RL socket (spindecoupler).
#   - Initialize the path sampling once (delegated to Robot_Script.rp_init).
#   - Wait for RL instructions:
#       * RESET: prepare to deliver the first observation.
#       * STEP: TP to current path sample, randomize target, wait N frames,
#               read observation via Agent.get_observation(), send it to RL,
#               advance (trial -> sample).
#       * FINISH: stop simulation.
#
# Notes:
#   - No RemoteAPI stepping is used; we count frames in sysCall_sensing().
#   - Target randomization and observation reuse the Agent's own methods.
#   - Path sampling and teleport are implemented in Robot_Script (rp_init/rp_tp).


import importlib
import logging
import os
import pkgutil
import shutil
import sys
import traceback

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
from plugins.agents import get_agent_factory


# Global variables for the simulation context
sim = None
agent = None
verbose = 1
agent_created = False
init_done = False
comm_init_done = False

# Config variables (values received from RL side)
path_alias = ""
sample_step_m = None
trials_per_sample = None
n_samples = None
n_extra_poses = None
delta_deg = None
base_pos_samples = None

# Other control variables
verbose = 3



def _autoload_agent_plugins(base_path):
    """Import all modules in plugins.agents so they self-register."""
    src_dir = os.path.join(base_path, "src")
    if os.path.isdir(src_dir) and src_dir not in sys.path:
        sys.path.insert(0, src_dir)         # for 'plugins.*'

    if base_path not in sys.path:
            sys.path.insert(0, base_path)   # para 'agents.*'
    try:
        pkg = importlib.import_module("plugins.agents")
        for finder, name, ispkg in pkgutil.iter_modules(pkg.__path__, "plugins.agents."):
            try:
                importlib.import_module(name)
                logging.info(f"[plugins] Imported: {name}")
            except Exception:
                logging.error(f"[plugins] Failed to import {name}")
                logging.debug(traceback.format_exc())
    except Exception:
        logging.error(f"Agent plugins autoload failed:\n{traceback.format_exc()}")


# -------------------------------
# ------- MAIN FUNCTIONS --------
# -------------------------------


def sysCall_init():
    """
    Called at the beginning of the simulation to configure logging and path setup. It alseo receives variable data from the RL side.
    """
    global sim, agent
    global init_done
    global robot_name, model_ids, params_scene, params_env, paths, file_id, model_name, verbose
    global save_scene, save_traj, action_times, obstacles_csv_folder, scene_to_load_folder
    global comms_port, ip_address
    global trials_per_sample, sample_step_m, n_samples, path_alias, n_extra_poses, delta_deg, place_obstacles_flag, random_target_flag, base_pos_samples

    sim = require('sim')    # type: ignore

    # Variables to get from agent_copp.py script
    comm_side = "agent"
    robot_name = "turtleBot"
    model_name = None
    model_ids = None
    comms_port = 49054
    ip_address = ""
    base_path = ""
    params_scene = {}
    params_env = {}
    verbose = 1
    scene_to_load_folder = ""
    obstacles_csv_folder = ""
    save_scene = None
    save_traj = None
    action_times = None
    path_alias = "/RecordedPath"
    sample_step_m = None
    trials_per_sample = None
    n_samples = None
    n_extra_poses = None
    delta_deg = None
    place_obstacles_flag = None
    random_target_flag = None
    base_pos_samples = None


    # Generate needed routes for logs and tf
    paths = utils.get_robot_paths(base_path, robot_name, agent_logs=True)
    file_id = utils.get_file_index(model_name, paths["tf_logs"], robot_name)
    utils.logging_config(paths["script_logs"], comm_side, robot_name, file_id, log_level=logging.INFO, verbose=verbose)

    # ----- Create Agent ------
    _autoload_agent_plugins(base_path)
    factory = get_agent_factory(robot_name)
    if factory is not None:
        agent = factory(sim, params_scene, params_env, paths, file_id, verbose, ip_address, comms_port=comms_port)
        logging.info(
            f"[plugins] Agent created via plugin for '{robot_name}'. "
            f"IP address: {ip_address}. "
            f"Comms port: {comms_port}"
        )
    else:   
        logging.info(f"[plugins] No agent plugin found for '{robot_name}'. ")

        # Fallback to hardcoded agents
        if robot_name == "turtleBot":
            agent = TurtleBotAgent(sim, params_scene, params_env, paths, file_id, verbose, ip_address, comms_port)
        elif robot_name == "burgerBot":
            agent = BurgerBotAgent(sim, params_scene, params_env, paths, file_id, verbose, ip_address, comms_port)
            agent.robot_baselink = agent.robot
        else:
            raise ValueError(f"Unknown robot name '{robot_name}' and no plugin found.")
        logging.info(f"Agent created via hardcoded class for '{robot_name}'. IP: {ip_address}. Comms port: {comms_port}.")
    
    init_done = True


def sysCall_thread():
    global sim, agent
    global init_done, comm_init_done
    global trials_per_sample, sample_step_m, n_samples, path_alias, n_extra_poses

    if not init_done: 
        pass

    comm_init_done = agent.start_communication()

    logging.info("Agent initialized and communication with RL side established")


    # ---- Configure scene and trajectory behavior -----
    agent.scene_to_load_folder = scene_to_load_folder
    agent.obstacles_csv_folder = obstacles_csv_folder
    agent.save_scene = save_scene
    agent.model_ids = model_ids
    agent.action_times = action_times
    agent.trials_per_sample = trials_per_sample

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


    # ----- Path sampling -----
    _robot_script = agent.handle_robot_scripts
    if base_pos_samples is None or base_pos_samples==[]:
        agent.path_handle = sim.getObject(path_alias)
        agent.path_pos_samples, agent.path_base_pos_samples = agent.sim.callScriptFunction('rp_init', _robot_script, n_samples, n_extra_poses, path_alias)
    
    else:
        logging.info(f"Positions have been provided by RL.")
        agent.grid_positions_flag = True

        # Valid positions
        agent.path_base_pos_samples = base_pos_samples
        logging.info(f"Total number of grid positions: {len(agent.path_pos_samples)}.")

        # We augment them by changing the orientation of the robot
        agent.path_pos_samples = agent.sim.callScriptFunction('augment_base_poses', _robot_script, agent.path_base_pos_samples, n_extra_poses, delta_deg)
    
    logging.info(f"Total number of scenarios to test: {len(agent.path_pos_samples)}.")

    logging.info(" ----- START EXPERIMENT ----- ")


def sysCall_sensing():
    """
    Called at each simulation step. Executes the agent's action and logs trajectories.
    """
    global sim, agent
    global verbose
    global comm_init_done
    global place_obstacles_flag, random_target_flag
    
    if agent and comm_init_done and not agent.finish_rec:
        # Loop for processing instructions from RL continuously until the agent receives a FINISH command.
        action = agent.agent_step_pv(place_obstacles_flag, random_target_flag)

        # If an action is received, execute it
        if action is not None:
            # With this check we avoid calling cmd_vel script repeteadly for the same action
            if agent.ts_received:
                if len(action)>0:
                    # agent.sim.callScriptFunction('cmd_vel', agent.handle_robot_scripts, action["linear"], action["angular"])
                    logging.info(f"Sample idx: {agent.current_sample_idx_pv}. Trial idx: {agent.current_trial_idx_pv}")
                    logging.info("Action has been predicted. Changing target and/or robot after action finishes")
                    # agent.sim.callScriptFunction('draw_path', agent.handle_robot_scripts, FIXED_SPEED, action["angular"], agent.colorID)
            agent.ts_received = False
        
        # Save current robot position for later saving it csv file
        # if agent.episode_idx >=1 and agent.save_traj:
        #     position = agent.sim.getObjectPosition(agent.robot_baselink, -1)
        #     agent.trajectory.append({"x": position[0], "y": position[1]})
        #     logging.debug(f"x pos: {position[0]}; y pos: {position[1]}")

    # FINISH command --> Finish the experiment
    if agent and agent.finish_rec:
        # Stop the simulation
        logging.info(" ----- END OF EXPERIMENT ----- ") # TODO This message appears multiple times before the simulation ends
        sim.stopSimulation()

    # Time tracking setup
    if agent and not agent.training_started:
        agent.initial_realTime = sim.getSystemTime()
        agent.initial_simTime = sim.getSimulationTime()
    