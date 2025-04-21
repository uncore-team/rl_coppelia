import csv
import datetime
import glob
import json
import logging
from logging.handlers import RotatingFileHandler
import math
import os
import re
import shutil
import subprocess
import sys
import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import psutil
from scipy.optimize import root_scalar, curve_fit
from scipy.interpolate import interp1d
from scipy.stats import linregress

AGENT_SCRIPT_COPPELIA = "/Agent_Script"
AGENT_SCRIPT_PYTHON = "common/agent_copp.py"


def initial_warnings(self):
    """
    Checks the provided arguments for missing values and sets default values if necessary.

    This function checks if certain required arguments are provided. If any of the arguments (`robot_name`, 
    `params_file`, `model_name`, `scene_path`) are missing or empty, the function prints the corresponding
    warning logs, and set some of them to their default values if neccesary.

    Args:
        args (Namespace): The command-line arguments passed to the script.

    Returns:
        
    """

    if hasattr(self.args, "robot_name") and not self.args.robot_name:
        logging.warning("WARNING: '--robot_name' was not specified, so default name 'burgerBot' will be used")
    
    if hasattr(self.args, "params_file") and not self.args.params_file:
        if self.args.command == 'train':
            self.args.params_file = os.path.join(self.base_path, "configs", "params_file.json")
        logging.warning("WARNING: '--params_file' was not specified, so default file will be used.")

    if hasattr(self.args, "model_name") and not self.args.model_name:  
        logging.warning("WARNING: '--model_name' is required for testing functionality. The testing experiment will use the last saved model.")

    if hasattr(self.args, "scene_path") and not self.args.scene_path:
        logging.warning(f"WARNING: '--scene_path' was not specified, so default one will be used: <robot_name>_scene.ttt. If this doesn't exist, it will use burgerBot_scene.ttt")


def logging_config(logs_dir, side_name, robot_name, experiment_id, log_level = logging.DEBUG, save_files = True, verbose = 0):
    """
    Configures the logging system for the application.

    Args:
        logs_dir (str): Directory where log files will be saved.
        side_name (str): Name of the process side (e.g., "agent" or "rl").
        robot_name (str): Name of the robot used in the experiment.
        experiment_id (int): Identifier for the current experiment session.
        log_level (optional): Logging level (default is logging.INFO).
        save_files (bool, optional): True for saving the log files.
        verbose (int, optional): True if we want to see logs in the terminal.

    Behavior:
        - Logs are displayed in the terminal and saved to a rotating log file.
        - Each log file has a maximum size of 50MB, and up to 4 backups are kept.
        - The log filename follows the format `{robot_name}_{side_name}_{experiment_id}.log`.
    """
    # Max size of each log file - 50 MB
    max_log_size = 0.05 * 1024 * 1024 * 1024 

    # Handler for managing log files
    log_file = os.path.join(logs_dir, f"{robot_name}_{side_name}_{experiment_id}.log") 
    rotating_handler = RotatingFileHandler(
        log_file, 
        maxBytes=max_log_size,  
        backupCount=4           # Keep the last 4 log files, remove the older ones
    )

    if save_files and verbose==2:
        log_handlers =[
            logging.StreamHandler(),  # Show through terminal
            rotating_handler          # Save in log files
        ]
    elif save_files and verbose !=2:
        # Set the log level for the rotating handler to WARNING
        rotating_handler.setLevel(logging.WARNING)  # TODO we are overriding here the input parameter, I need to check this
        log_handlers =[rotating_handler]
    else:
        log_handlers =[logging.StreamHandler()]

    logging.basicConfig(
        level=log_level,  
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=log_handlers
    )


def get_last_model(models_path):
    """
    Gets the last modified model (so it should be the last trained model) inside the models folder

    Args:
        models_path (str): Path to the models folder.
    
    Returns:
        last_model_name (str): Name of the last modified model.
        last_model_path (str): Path to the last modified model.
    
    """
    if os.path.exists(models_path):
        # Get last modified file
        files = sorted(os.listdir(models_path), key=lambda x: os.path.getmtime(os.path.join(models_path, x)))
        if files:
            last_model_name = files[-1]
            last_model_path = os.path.join(models_path, last_model_name)
            return last_model_name, last_model_path
            
        else:
            logging.critical("No model files found in the models folder.")
            sys.exit()
    else:
        logging.critical("Models folder does not exist, testing cannot be done.")
        sys.exit()


def get_file_index(args, tf_path, robot_name):
    """
    In training mode: retrieves the next available id for the files that will be generated by the current execution of the program.

    In testing mode: retrieves the model ID of the specified model followed by a timestamp: <model_ID>_<timestamp>.

    Args:
    #TODO model_name/args
        tf_path (str): Path to the TensorBoard log directory.
        robot_name (str): Name of the robot, used to identify log files.

    Returns:
        index (str): In training mode, the highest indez found in the existing TensorBoard folder with logs named '{robot_name}_tflogs_<index>' 
            , and it will be used for creating the files associated with the current execution. In testing mode, the model ID to be tested 
            followed by the current timestamp, and it will be used just for naming the logs.
    """
    if not hasattr(args, "model_name") and not isinstance(args, str):
        tf_name = f"{robot_name}_tflogs"
        max_index = 0
        for path in glob.glob(os.path.join(tf_path, f"{glob.escape(tf_name)}_[0-9]*")):
            file_name = path.split(os.sep)[-1]
            ext = file_name.split("_")[-1]
            if tf_name == "_".join(file_name.split("_")[:-1]) and ext.isdigit() and int(ext) > max_index:
                max_index = int(ext)
        index = str(max_index + 1)
    elif isinstance(args, str):
        model_name = args
        model_id = model_name.rsplit("_")[2]
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        index = f"{model_id}_{timestamp}"
    else:
        model_id = args.model_name.rsplit("_")[2]
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        index = f"{model_id}_{timestamp}"

    return index


def get_next_model_name(path, robot_name, next_index, callback_mode = False):
    """
    Generates a new model/caollback filename based on the training stage.

    Args:
        path (str): Directory where the model/callback should be saved.
        robot_name (str): Name of the robot used in training.
        next_index (str): The next model/callback index number.
        callback_mode (bool, optional): Whether the model is being saved during training callbacks (default: False).

    Returns:
        to_save_path (str): Full path for saving the model.

    Behavior:
        - If 'callback_mode' is True, the filename format is '{robot_name}_callbacks_<index>'.
        - Otherwise, the filename format is '{robot_name}_model_<index>'.
    """
    # Get the right path to save the trained model
    if callback_mode:
        new_model_name = f"{robot_name}_callbacks_{int(next_index):01d}"
    else:
        new_model_name = f"{robot_name}_model_{int(next_index):01d}"
    to_save_path = os.path.join(path, new_model_name)

    return to_save_path, new_model_name


def get_robot_paths(base_dir, robot_name, just_agent_logs = False):
    """
    Generate the paths for working with a robot, and create the needed folders if they don't exist

    Args:
        base_dir (str): base path where the subfolders will be craeted
        robot_name (str): name of the robot.
        just_agent_logs (bool, optional): Only for generating the paths needed for the agent.

    Returns:
        paths (dict): Dictionary with the given paths.
    """
    if not just_agent_logs:
        paths = {
            "models": os.path.join(base_dir, "robots", robot_name, "models"),
            "callbacks": os.path.join(base_dir, "robots",robot_name, "callbacks"),
            "tf_logs": os.path.join(base_dir, "robots",robot_name, "tf_logs"),
            "script_logs": os.path.join(base_dir, "robots",robot_name, "script_logs", "rl_logs"),
            "testing_metrics": os.path.join(base_dir, "robots",robot_name, "testing_metrics"),
            "training_metrics": os.path.join(base_dir, "robots",robot_name, "training_metrics"),
            "parameters_used": os.path.join(base_dir, "robots",robot_name, "parameters_used"),
            "configs": os.path.join(base_dir, "configs")
        }
    else:
        paths = {
            "tf_logs": os.path.join(base_dir, "robots",robot_name, "tf_logs"),
            "script_logs": os.path.join(base_dir, "robots",robot_name, "script_logs", "agent_logs")
        }

    # Create the folders if they don't exist
    for path in paths.values():
        os.makedirs(path, exist_ok=True)

    return paths


def find_coppelia_path():
    """
    Attempts to locate the CoppeliaSim installation directory automatically.
    
    The function first checks if CoppeliaSim is available in the system PATH. 
    If not found, it searches common installation directories.
    If still not found, it checks for an environment variable 'COPPELIA_PATH'.
    
    Returns:
        str: The absolute path to the CoppeliaSim installation directory if found, otherwise None.
    """
    # Check if CoppeliaSim is in PATH
    coppelia_exe = shutil.which("coppeliaSim")
    if coppelia_exe:
        return os.path.dirname(coppelia_exe)
    
    # Search in common installation directories
    common_paths = [
        os.path.expanduser("~/Documents"),  # Search in Documents folder
        os.path.expanduser("~/Downloads"),  # Search in Downloads folder
        os.path.expanduser("~/devel"),
        "/opt", "/usr/local", "/home"     # Common system directories
    ]
    
    for path in common_paths:
        for root, dirs, files in os.walk(path):
            if "coppeliaSim.sh" in files:  # CoppeliaSim executable in Linux
                return root
    
    # Check environment variable
    return os.getenv("COPPELIA_PATH", None)


def stop_coppelia_simulation (sim):
    """
    Check if Coppelia simulation is running and, in that case, it stops the simulation.

    Args:
        sim: CoppeliaSim object.
    """
    # Check simulation's state before stopping it
    if sim.getSimulationState() != sim.simulation_stopped:
        sim.stopSimulation()

        # Wait until the simulation is completely stopped
        while sim.getSimulationState() != sim.simulation_stopped:
            time.sleep(0.1)


def is_coppelia_running():
    """
    Check if Coppelia has been already executed.

    Returns:
        bool : True if CoppeliaSim is running.
    """

    # Check if CoppeliaSim process is running
    for process in psutil.process_iter(attrs=['name']):
        if "coppeliaSim" in process.info['name']:
            return True
    return False


def is_scene_loaded(sim, scene_path):
    """
    Check if the desired scene is loaded.
    Args:
        sim: CoppeliaSim object.
        scene_path (str): Path to the scene

    Returns:
        bool: True if the input scene has been loaded.
    """

    # If Coppelia is running, check if the input scene is loaded
    try:
        current_scene = sim.getStringParam(sim.stringparam_scene_path_and_name)
        if os.path.abspath(current_scene) == os.path.abspath(scene_path):
            return True
        else:
            return False
    except Exception as e:
        logging.critical(f"Error while getting scene name: {e}")
        sys.exit()


def update_and_copy_script(sim, base_path, args, params_env, comms_port):
    """
    Updates and the agent script of the CoppeliaSim scene with the content of the 'rl_coppelia/agent_copp.py'.

    This function loads the agent script from a specified path, updates certain variables 
    in the script (such as robot name, base path, communication port, and environment parameters), 
    and sends the updated script content back to the CoppeliaSim scene.

    Args:
        sim (object): The CoppeliaSim simulation object that provides access to the simulation environment.
        base_path (str): The base directory where the script is located.
        args #TODO
          (str): The name of the robot to be used in the simulation.
        params_env (dict): A dictionary containing environment parameters to be passed into the script.
        comms_port (int): The communication port number used for communication with the robot.

    Returns:
        bool: True if the script was successfully updated and copied to CoppeliaSim, False otherwise.

    Raises:
        Exception: If an error occurs during the script update or copying process.
    """
    # Load Agent script
    agent_object = sim.getObject(AGENT_SCRIPT_COPPELIA)
    agent_script_handle = sim.getScript(1, agent_object)

    # Read Agent_script.py  
    agent_script_path = os.path.join(base_path, "src", AGENT_SCRIPT_PYTHON)
    logging.info(f"Copying content of {agent_script_path} inside the scene in {AGENT_SCRIPT_COPPELIA}")

    try:
        with open(agent_script_path, "r") as file:
            script_content = file.read()

        # Dictionary with variables to update
        if not hasattr(args, "model_name"):
            args.model_name = None

        replacements = {
            "robot_name": args.robot_name,
            "model_name": args.model_name,
            "base_path": base_path,
            "comms_port": comms_port,
            "verbose": args.verbose,
            "testvar": comms_port+1,
        }

        # Update standard variables
        for var, new_value in replacements.items():
            if isinstance(new_value, str):
                script_content = re.sub(rf'{var}\s*=\s*["\'].*?["\']', f'{var} = "{new_value}"', script_content)
                script_content = re.sub(rf'{var}\s*=\s*None', f'{var} = "{new_value}"', script_content)
            else:
                script_content = re.sub(rf'{var}\s*=\s*\d+', f'{var} = {new_value}', script_content)
                script_content = re.sub(rf'{var}\s*=\s*None', f'{var} = {new_value}', script_content)

        # Format the 'params_env' dictionary
        params_str = "{"
        for key, value in params_env.items():
            if isinstance(value, bool):
                # Convert booleans to the right format (True/False)
                params_str += f'\n    "{key}": {"True" if value else "False"},'
            elif isinstance(value, str):
                # Add quotation marks for strings
                params_str += f'\n    "{key}": "{value}",'
            else:
                # Numbers and other types
                params_str += f'\n    "{key}": {value},'
        params_str += "\n}"

        # Replace the script content with the formatted dictionary.
        script_content = re.sub(r'params_env\s*=\s*\{\}', f'params_env = {params_str}', script_content)

        # Save the file with the changes
        # with open(agent_script_path, "w") as file:
        #     file.write(script_content)

        # Send updated script content to script C in CoppeliaSim
        sim.setScriptText(agent_script_handle, script_content)
        logging.info("Script updated successfully in CoppeliaSim.")
        return True

    except Exception as e:
        logging.error(f"Something happened while trying to update the content of the script inside Coppelia's scene: {e}")
        sys.exit()


def is_port_in_use(port):
    """
    Verify if a port is being used or in LISTEN state.
    """
    for conn in psutil.net_connections(kind="inet"):
        if conn.laddr.port == port and conn.status in ("LISTEN", "ESTABLISHED"):
            return True  # The port is busy
    return False  # The port is free


def find_next_free_port(start_port = 49054):
    """
    Finds the next available port starting from the given port.

    This function checks whether a port and the next consecutive port are available for use. 
    If both ports are available, it returns the first available port. If not, it increments 
    the port by 2 and continues the search. If no available ports are found within a reasonable 
    range, the function will log an error and exit.

    Args:
        start_port (int): The starting port number to search from. Default is 49054.

    Returns:
        int: The next available port.

    Raises:
        SystemExit: If no available ports are found after checking a range of 50 ports.

    Notes:
        The function checks pairs of ports, as Coppelia uses two consecutive ports for communication 
        (by default, `zmqRemoteApi.rpcPort` and `zmqRemoteApi.cntPort`).
    """
    next_port = start_port
    while True:
        if not is_port_in_use(next_port) and not is_port_in_use(next_port + 1):
            logging.info(f"Next avaliable port to be used: {next_port}")
            return next_port 
        next_port += 2  # If it's busy, let's try with the next pair of ports
        if next_port - start_port > 50:
            logging.error(f"No ports avaliable for communication. Last port that was attempted to connect was {next_port}")
            sys.exit()
   

def create_discs_under_target(sim, params_env):
    """
    Creates three disc shapes in CoppeliaSim, assigns them as children of the Target object, 
    sets their colors, positions them at different heights, and gives them specific names.
    Removes any existing child discs before creating new ones.
    
    Args:
        sim (coppeliaObject): Simulation object for CoppeliaSim.
        params_env (dict): Parameters of the environment - radius of the discs.

    Returns:
        list: A list containing the handles of the three created discs.
    """
    # Get target handle
    target_handle = sim.getObject("/Target")

    # Remove any existing child discs
    child_objects = sim.getObjectsInTree(target_handle, sim.handle_all, 1)  # Get direct children
    for child in child_objects:
        obj_type = sim.getObjectType(child)
        if obj_type == sim.object_shape_type:  # Ensure it's a shape before deleting
            sim.removeObject(child)

    # Disc properties: name, color (R, G, B), relative Z position
    disc_properties = [
        ("Target_disc_1", [0, 0, 1], 0.002, params_env["reward_dist_1"]),  # Blue
        ("Target_disc_2", [1, 0, 0], 0.004, params_env["reward_dist_2"]),  # Red
        ("Target_disc_3", [1, 1, 0], 0.006, params_env["reward_dist_3"])   # Yellow
    ]

    disc_handles = []

    for name, color, z_offset, radius in disc_properties:
        # Create a disc with minimal thickness
        disc_handle = sim.createPrimitiveShape(sim.primitiveshape_disc, [radius * 2, radius * 2, 0.01], 0)

        # Set the disc's alias (name in the scene)
        sim.setObjectAlias(disc_handle, name)

        # Set the disc as a child of the target object
        sim.setObjectParent(disc_handle, target_handle, True)

        # Adjust the relative position (lifting it in the Z axis)
        sim.setObjectPosition(disc_handle, target_handle, [0, 0, z_offset])

        # Set the color (ambient diffuse component)
        sim.setShapeColor(disc_handle, None, sim.colorcomponent_ambient_diffuse, color)

        # Store the handle in the list
        disc_handles.append(disc_handle)

    return disc_handles  # Return the handles of the created discs


def start_coppelia_and_simulation(base_path, args, params_env, comms_port):
    """
    Run CoppeliaSim if it's not already running and open the scene if it's not loaded.

    Args:
        base_path (str): Path of the base directory.
        args: It will use two of the input arguments: robot_name and no_gui.
            robot_name (str): NAme of the current robot.
            no_gui_option (bool): If true, it will initialize Coppelia without its GUI.
        params_env (dict): A dictionary containing environment parameters.
        comms_port (int): The communication port number used for communication with the robot.

    Returns:
        sim: CoppeliaSim object in case that the program is running and the scene is loaded successfully.
    """
    process = None
    zmq_port = 23000    # Default port for zmq communication
    ws_port = 23050     # Default for websocket communication

    # CoppeliaSim path
    coppelia_path = find_coppelia_path()
    coppelia_exe = os.path.join(coppelia_path, 'coppeliaSim.sh')

    # Scene path
    if args.scene_path is None:
        try:
            args.scene_path = os.path.join(base_path, "scenes", f"{args.robot_name}_scene.ttt")
        except:
            args.scene_path = os.path.join(base_path, "scenes/burgerBot_scene.ttt")

    # Verify if CoppeliaSim is running
    # TODO Check that when we open several instances of CoppeliaSim with the GUI, only
    # one will be preserved if the screen powers off automatically for saving energy.
    # So please use the no_gui mode for the moment if you are leaving the PC.
    if args.dis_parallel_mode:
        if not is_coppelia_running():
            logging.info("Initiating CoppeliaSim...")
            if args.no_gui:
                process = subprocess.Popen(["gnome-terminal", "--",coppelia_exe, "-h"])
            else:
                process = subprocess.Popen(["gnome-terminal", "--",coppelia_exe])
        else:
            logging.info("CoppeliaSim was already running")
    else:
        logging.info("Initiating a new CoppeliaSim instance...")
        zmq_port = find_next_free_port(zmq_port)    
        ws_port = find_next_free_port(ws_port)
        if args.no_gui:
            # process = subprocess.Popen(["gnome-terminal", "--",coppelia_exe, "-h", f"-GzmqRemoteApi.rpcPort={zmq_port}", f"-GwsRemoteApi.port={ws_port}"])
            process = subprocess.Popen([
                "gnome-terminal", 
                "--", 
                "bash", "-c", 
                f"{coppelia_exe} -h -GzmqRemoteApi.rpcPort={zmq_port} -GwsRemoteApi.port={ws_port}; exec bash"
            ])
        else:
            # process = subprocess.Popen(["gnome-terminal", "--",coppelia_exe, f"-GzmqRemoteApi.rpcPort={zmq_port}", f"-GwsRemoteApi.port={ws_port}"])
            process = subprocess.Popen([
                "gnome-terminal", 
                "--", 
                "bash", "-c", 
                f"{coppelia_exe} -GzmqRemoteApi.rpcPort={zmq_port} -GwsRemoteApi.port={ws_port}; exec bash"
            ])
            
    # Wait for CoppeliaSim connection
    try:
        logging.info("Waiting for connection with CoppeliaSim...")
        client = RemoteAPIClient(port=zmq_port)
        sim = client.getObject('sim')
    except Exception as e:
        logging.error(f"It was not possible to connect with CoppeliaSim: {e}")
        if process:
            process.terminate()
        return False

    logging.info("Connection established with CoppeliaSim")

    # Check if scene is loaded
    if is_scene_loaded(sim, args.scene_path):
        logging.info("Scene is already loaded, simulation will be stopped in case that it's running...")
        stop_coppelia_simulation(sim)
    else:
        logging.info(f"Loading scene: {args.scene_path}")
        sim.loadScene(args.scene_path)
        logging.info("Scene loaded successfully.")

    # Create target's discs # TODO
    create_discs_under_target(sim, params_env)

    # Update code inside Coppelia's scene
    update_and_copy_script(sim, base_path, args, params_env, comms_port)

    # Start the simulation
    sim.startSimulation()
    logging.info("Simulation started")

    # For updating specific variable --> DEPRECATED
    #sim.setStringSignal("robotName", robot_name)
    #sim.setStringSignal("basePath", base_path)
    #logging.info("Variables sent to Coppelia scene")

    return sim


def get_new_coppelia_pid(before_pids):
    """
    Detect the PID of the current CoppeliaSim instance. This function will allow
    us to close the CoppeliaSim instance that has finished its training.
    """
    time.sleep(4)  # Wait for the process to be created
    after_pids = {proc.pid: proc.name() for proc in psutil.process_iter(['pid', 'name'])}
    new_pids = set(after_pids.keys()) - set(before_pids.keys())

    copp_pids=[]

    for pid in new_pids:
        if "coppelia" in after_pids[pid].lower():
            copp_pids.append(pid)
    if copp_pids:       
        logging.info(f"New CoppeliaSim processes detected: PID {copp_pids}")
        return copp_pids
    
    logging.warning("Error: No new Coppelia process detected.")
    return None


def _get_default_params ():
    """
    Private function for setting the different parameters to their default values, in case that reading the json fails.

    Args: None

    Return:
        dicts: Three dictionaries with the parameters of the environment, the training and the testing process.
    """
    params = {
        "params_env": {
            "var_action_time_flag": False,
            "fixed_actime": 1.0,
            "bottom_actime_limit": 0.2,
            "upper_actime_limit": 3.0,
            "bottom_lspeed_limit": 0.05,
            "upper_lspeed_limit": 0.5,
            "bottom_aspeed_limit": -0.5,
            "upper_aspeed_limit": 0.5,
            "finish_episode_flag": False,
            "dist_thresh_finish_flag": 0.5,
            "obs_time": False,
            "reward_dist_1": 0.25,
            "reward_1": 25,
            "reward_dist_2": 0.05,
            "reward_2": 50,
            "reward_dist_3": 0.015,
            "reward_3": 100,
            "max_count": 400,
            "max_time": 80,
            "max_dist": 2.5,
            "finish_flag_penalty": -10,
            "overlimit_penalty": -10,
            "crash_penalty": -20,
            "max_crash_dist": 0.1

        },
        "params_train": {
            "sb3_algorithm": "SAC",
            "policy": "MlpPolicy",
            "total_timesteps": 300000,
            "callback_frequency": 10000,
            "n_training_steps": 2048
        },
        "params_test": {
            "sb3_algorithm": "",
            "testing_iterations": 50
        }
    }
    return params["params_env"], params["params_train"], params["params_test"]


def load_params(file_path):
    """
    Load the configuration file as a dictionary.

    Args:
        file_path (str): Path to the JSON configuration file.

    Returns:
        params_env (dict): Parameters for configuring the environment.
        params_train (dict): Parameters for configuring the training process.
        params_test (dict): Parameters for configuring the testing process.
    """
    try:
        with open(file_path, 'r') as f:
            params_file = json.load(f)
            if params_file :
                params_env = params_file["params_env"]
                params_train = params_file["params_train"]
                params_test = params_file["params_test"]
                logging.info(f"Configuration loaded successfully from {file_path}.")
                return params_env, params_train, params_test
            else:
                logging.error("Failed to load configuration. Default values will be used")
                return _get_default_params()
            
    except Exception as e:
        logging.error(f"Error loading configuration file: {e}")
        return _get_default_params()
    
    
def get_output_csv(model_name, metrics_path, train_flag=True):
    """
    Get the path to to csv file that will be generated for storing the training/inference metrics. The name of the file
    will be unique, as it makes use of the timestamp.

    Args:
        model_name (str): Name of the model file, as it will be used for identifying the csv file.
        metrics_path (str): Path to store the csv files with the obtained metrics.
        train_flag (bool): True if the script has been executed in training mode, False in case of running a test. True by default.

    Return:
        output_csv_path (str): Path to the new csv file.
    """
    # Get current timestamp so the metrics.csv file will have an unique name
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Get name and return the path to the csv file
    if train_flag:
        output_csv_name = f"{model_name}_train_{timestamp}.csv"
    else:
        output_csv_name = f"{model_name}_test_{timestamp}.csv"
    output_csv_path = os.path.join(metrics_path, output_csv_name)
    return output_csv_name, output_csv_path


def update_records_file (file_path, exp_name, start_time, end_time, other_metrics):
    """
    Function to update the train or test record file, so the user can track all the training or testing attempts.

    Args:
        file_path (str): Path to the csv file which stores the training/testing records.
        exp_name (str): ID of the experiment.
        start_time (timestamp): Time at the beggining of the experiment.
        end_time (timestamp): Time at the end of the experiment.
        other_metrics (dic): Dictionary with other metrics that the user wants to record.
    """
    # Create the dictionary with the data to be saved in the CSV file
    data = {
        "Exp_id": exp_name,
        "Start_time": datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S'),
        "End_time": datetime.datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S'),
        "Duration": (end_time - start_time)/3600,  # Training hours
        **other_metrics  # Add final metrics (loss, etc.)
    }

    # Create the header if the file does not exist
    # Define the CSV header
    headers = list(data.keys())

    try:
        with open(file_path, mode="r") as f:
            pass
    except FileNotFoundError:
        with open(file_path, mode="w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)  # Write the headers

    # Write data
    with open(file_path, mode="a", newline='') as f:
        values = [data.get(header, '') for header in headers]
        writer = csv.writer(f)
        writer.writerow(values)

    logging.info(f"Record file has been updated in {file_path}")


def copy_json_with_id(source_path, destination_dir, file_id):
    """
    Copies the given JSON file to a specified directory and appends a file_id to its name.

    Args:
        source_path (str): Path to the original JSON file.
        destination_dir (str): Directory where the file should be copied to.
        file_id (str): The file ID to append to the copied file's name.
    
    Returns: None
    """
    # Ensure the destination directory exists
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Get the filename from the source path
    filename = os.path.basename(source_path)
    name, ext = os.path.splitext(filename)

    # Create the new filename with file_id
    new_filename = f"{name}_model_{file_id}{ext}"

    # Create the destination path
    destination_path = os.path.join(destination_dir, new_filename)

    # Copy the file
    shutil.copy(source_path, destination_path)

    logging.info(f"A copy of the parameters file has been copied to {destination_path}")

import csv


def get_algorithm_for_model(model_name, csv_path):
    """
    Searches for a row in the CSV file where the first column matches the given model name,
    and returns the value in the "Algorithm" column for that row. Doing this we can be sure
    that we are testing the model using the same algorithm that was used for training it.

    Args:
        model_name (str): The model name to search for.
        csv_path (str): The path to the CSV file.

    Returns:
        alg_name (str): The value from the "Algorithm" column corresponding to the row where 
                     the model name is found
    """
    with open(csv_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        # Get the key of the first column (assumed to contain the model names)
        first_col = reader.fieldnames[0]
        for row in reader:
            if row[first_col] == model_name:
                return row.get("Algorithm")
            
    logging.error("There was an error while checking the algorithm used for training the model")
    raise ValueError(f"Model '{model_name}' not found in CSV file '{csv_path}'")


def get_params_file(paths, args):
    """
    Retrieves the path to the parameter configuration file associated with a given model.

    This function checks if a model name is provided. If not, it retrieves the latest model name from the 
    specified models directory. It then extracts the model ID from the model name and searches for the 
    corresponding configuration file (ending with "_model_<id>.json") in the parameters directory.

    Args:
        paths (dict): A dictionary containing paths, including 'models' (for model files) 
                      and 'parameters_used' (for parameter configuration files).
        args (argparse.Namespace): The arguments passed to the function, including the optional 'model_name'.

    Returns:
        str: The full path to the parameter configuration file for the corresponding model.

    Raises:
        SystemExit: If no model ID is found in the model name or if the corresponding parameter file 
                    is not found in the parameters directory.
    """
    models_path = paths["models"]
    parameters_used_path = paths["parameters_used"]

    # Check if a model name was provided by the user
    if args.model_name is None:
        model_name, _ = get_last_model(models_path)
    else:
        model_name = args.model_name
    
    # Extract the model_id from the model name
    match = re.search(r"model_\d+", model_name)
    if not match:
        logging.critical(f"No model files ending with 'model_\d' found in the {models_path} folder.")
        sys.exit()  
    
    model_id = match.group(0)  # "model_<id>"
    
    # Search the configuration file that ends with the corresponding "_model_<id>.json"
    for file in os.listdir(parameters_used_path):
        if file.endswith(f"{model_id}.json"):
            params_file_path = os.path.join(parameters_used_path, file)  # Saves the whole path
            logging.info(f"The parameter file that will be used for testing is {params_file_path}")
            return params_file_path
    
    logging.critical(f"No configuration file ending with {model_id}.json found in the {parameters_used_path} folder.")
    sys.exit()  


def auto_create_param_files(base_params_file, output_dir, start_value, end_value, increment):
    """
    Creates parameter files with incrementing fixed_actime values.
    First cleans the output directory of any existing JSON files, preserving CSV files.
    
    Args:
        base_params_file (str): Path to the base parameters file.
        output_dir (str): Directory to save the generated parameter files.
        start_value (float): Starting value for fixed_actime.
        end_value (float): Ending value for fixed_actime.
        increment (float): Increment value for fixed_actime.
        
    Returns:
        list: List of paths to the generated parameter files.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Clean the directory by removing all existing files
    for file in os.listdir(output_dir):
        if file.lower().endswith('.json'):
            file_path = os.path.join(output_dir, file)
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                    logging.info(f"Removed existing JSON file: {file_path}")
                except Exception as e:
                    logging.error(f"Error removing JSON file {file_path}: {e}")
    
    # Read base parameters file
    with open(base_params_file, 'r') as f:
        base_params = json.load(f)
    
    param_files = []
    current_value = start_value
    
    # Create parameter files with incrementing fixed_actime values
    while current_value <= end_value:
        # Update fixed_actime value
        base_params["params_env"]["fixed_actime"] = round(current_value, 4)  # Round to avoid floating point precision issues
        
        # Create output file name
        output_file = os.path.join(output_dir, f"params_actime_{current_value:.4f}.json")
        
        # Write parameters to file
        with open(output_file, 'w') as f:
            json.dump(base_params, f, indent=2)
        
        param_files.append(output_file)
        current_value += increment
    
    return param_files


def auto_run_mode(args, mode, file = None, model_id = None, no_gui=True):
    """
    Runs the training process using a specified parameter file. This function executes the training
    through a subprocess, optionally suppressing the GUI and enabling parallel mode.

    Args:
        args (argparse):
            robot_name (str).
            dis_parallel_mode (bool).
            model_ids (int).
        mode (str): For choosing between different possible modes.
        file (str, optional): The parameter file to use for the training modes.
        no_gui (bool): A flag to suppress the GUI during training/testing. Default is True.

    Returns:
        str: The name of the parameter file used for training.
        str: The status of the training ("Success" or "Failed").
        float: The duration of the training in hours.
    """
    model_name = ""

    if mode != "sampling_at" and mode != "auto_training" and mode != "auto_testing":
        logging.critical(f"ERROR: the specified training mode doesn't exist. The provided mode was {mode}. Execution will end.")
        sys.exit()

    if file is not None:
        logging.info(f"Starting training with parameter file: {file}")
    else:
        logging.info(f"Starting testing with model ids {args.model_ids}")

    if mode == "sampling_at":
        fixed_actime = os.path.basename(file).split('_')[-1].replace('.json', '')

    # Ger current opened processes in the PC so later we can know which ones are the Coppelia new ones.
    before_pids = {proc.pid: proc.name() for proc in psutil.process_iter(['pid', 'name'])}

    # Record the start time of the training
    start_time = time.time()

    # Command to run the training script with the specified parameter file
    if mode != "auto_testing":  # Just for training modes
        cmd = ["rl_coppelia", "train"]

        # Add the parameter file to the command
        cmd.extend(["--params_file", file])

    else:   # If it's for auto testing mode
        cmd = ["rl_coppelia", "test"]

        # Add the model name to the command
        model_name = args.robot_name + "_model_" + str(model_id)
        cmd.extend(["--model_name", model_name])

    # Add the flag to suppress the GUI if specified
    if no_gui:
        cmd.append("--no_gui")

    # Add the disable parallel mode flag if you want to run the training sequentially
    if args.dis_parallel_mode:
        cmd.append("--dis_parallel_mode")

    # Add the robot name
    if args.robot_name:
        cmd.extend(["--robot_name", args.robot_name])

    # Add the robot name
    if args.verbose:
        cmd.extend(["--verbose", str(args.verbose)])

    logging.info(f"CMD: {cmd}")

    try:
        # Run the command as a subprocess and capture the output
        process = subprocess.Popen(cmd, stderr=subprocess.STDOUT, text=True)        

        # Get the id of the new process.
        coppelia_pid = get_new_coppelia_pid(before_pids)
        
        # Wait for process to complete.
        process.communicate()
        
        # Check the result of the training/testing process
        if (process.returncode != 0 and process.returncode is not None) or process.stderr is not None:
            status = "Failed"
            if file is not None:
                logging.error(f"Error in process with file {os.path.basename(file)}: {process.stderr}")
            else:
                logging.error(f"Error in process with model name {model_name}: {process.stderr}")
        else:
            status = "Success"
        
        if coppelia_pid:
            try:
                for pid in coppelia_pid:
                    coppelia_proc = psutil.Process(pid)
                    logging.info(f"Closing CoppeliaSim (PID {pid})...")
                    coppelia_proc.terminate()
                    coppelia_proc.wait(timeout=10)
                    logging.info(f"CoppeliaSim (PID {pid}) closed.")
            except psutil.NoSuchProcess:
                logging.warning("CoppeliaSim process didn't exist anymore when trying to close it.")
            except psutil.TimeoutExpired:
                logging.warning("CoppeliaSim didn't answer, forcing ending.")
                coppelia_proc.kill()

    except Exception as e:
        status = "Exception"
        if file is not None:
            logging.error(f"Exception in process with file {os.path.basename(file)}: {e}")
        else:
            logging.error(f"Exception in process with model name {model_name}: {e}")
    
    # Record the end time and calculate the duration of the training
    end_time = time.time()
    process_duration = (end_time - start_time) / 3600.0

    if file is not None:
        logging.info(f"Finished training with parameter file: {os.path.basename(file)} - {status}, duration: {process_duration:.3f} hours")
    else:
        logging.info(f"Finished testing the model: {model_name} - {status}, duration: {process_duration:.3f} hours")
    
    # Return the results:
    if mode == "sampling_at":
        # file name, fixed action time, training status, and duration
        return os.path.basename(file), fixed_actime, status, process_duration

    elif mode == "auto_training":
        # file name, training status, and duration
        return os.path.basename(file), status, process_duration

    elif mode == "auto_testing":
        # model name, testing status, and duration
        return model_name, status, process_duration
        

def create_next_auto_test_folder(base_path):
    # Pattern to search: "auto_test_XX"
    pattern = re.compile(r'auto_test_(\d+)')
    
    # Get all the folders with that pattern
    existing_folders = [f for f in os.listdir(base_path) if pattern.match(f) and os.path.isdir(os.path.join(base_path, f))]
    
    # Extract the ids
    indices = [int(pattern.match(f).group(1)) for f in existing_folders]
    
    # Get next id
    next_index = max(indices, default=0) + 1
    new_folder_name = f'auto_test_{next_index:02d}'
    new_folder_path = os.path.join(base_path, new_folder_name)
    
    # Create the new folder
    os.makedirs(new_folder_path, exist_ok=True)
    logging.info(f'Carpeta creada: {new_folder_path}')
    return new_folder_path, new_folder_name


def get_data_for_spider(csv_path, args, column_names):
    """
    Searches for rows in a CSV file where the first column contains an experiment name 
    that ends with a given ID (formatted as '<robot_name>_model_<id>'), and returns the 
    mean of specific columns' data for the matched rows for each ID in the provided list.

    Args:
    - csv_path (str): The file path to the CSV file containing the experiment data.
    - args (argparse.Namespace): An object containing the arguments passed via argparse.
      - args.robot_name (str): The name of the robot used to filter the experiment names.
      - args.ids (list of int): A list of IDs to match at the end of the experiment name.
    - column_names (list): A list of column headers (strings) whose data will be averaged 
      across all matching rows for each ID.

    Returns:
    - dict: A dictionary where the keys are the IDs, and the values are pandas Series 
      containing the mean values of the specified columns for the matching rows of each ID. 
      If no rows are found for a specific ID, the value for that ID will be None.
    
    Example:
    - If the CSV has rows like 'robot1_model_34', 'robot2_model_34', 
      and 'robot1_model_134', and you call the function with robot_name='robot1', ids=[34, 134],
      it will return a dictionary with the mean values for the matching rows for each ID.
    """
    # Cargar el CSV en un DataFrame de pandas
    try:
        df = pd.read_csv(csv_path)

    except Exception as e:
        logging.error(f"No csv file was found. Exception: {e}")
        sys.exit()
    
    data_to_extract = {}

    df_filtered = df[df.iloc[:, 0].notna()]
    # Process each ID in the args.model_ids list
    for id in args.model_ids:
        # Search rows in the first column which finish with the provided ID
        print(f'{args.robot_name}_model_{id}')
        

        filter = df_filtered.iloc[:, 0].apply(lambda x: x.startswith(f'{args.robot_name}_model_{id}'))
        filtered_rows = df_filtered[filter]
        
        # If no row is found, then assign None
        if filtered_rows.empty:
            data_to_extract[id] = None
        else:
            # Select the desired columns and calculte the mean
            data = filtered_rows[column_names].mean(axis=0)
            data_to_extract[id] = data
    
    return data_to_extract


def process_spider_data (df, tolerance=0.05):
    """
    Extracts data from the train and test dataframes and normalizes those metrics for radar chart visualization, ensuring:
    - Min-Max Scaling for standard metrics.
    - Inverse scaling for loss metrics (Min-Max on absolute value).
    - Inverse scaling for `rollout/ep_len_mean` to prioritize lower values.
    - A tolerance is applied to avoid exact 0 or 1 in the normalized values.

    Args:
    - df (DataFrame): Dataframe with data for each ID as pandas.Series.
    - tolerance (float): Percentage tolerance applied to prevent normalization from reaching 0 or 1.

    Returns:
    - data_list (list of lists): Normalized metric values for each ID.
    - names (list): List of experiment names formatted as "T_<action_time>".
    - labels (list): List of metric names.
    """
    data_list = []
    names = []

    # Extract metric labels excluding "Action time (s)"
    labels = df.drop(columns=["Action time (s)"]).columns.tolist()

    # Separate the different metrics depending on how they work:
    negative_metrics = [col for col in labels if "actor_loss" in col.lower()]  # More negative values --> Better
    min_metrics = [col for col in labels if any(metric in col.lower() for metric in ["time", "critic_loss", "ep_len_mean"])]    # Smaller values (closer to 0) --> Better
    max_metrics = [col for col in labels if col not in negative_metrics + min_metrics]    # Bigger values --> Better

    # Normalize data
    df_normalized = df.copy()

    # --- max_metrics = ['rollout/ep_rew_mean', 'Avg reward']
    min_values = df[max_metrics].min()
    max_values = df[max_metrics].max()
    ranges = max_values - min_values
    # Apply Min-Max Scaling with tolerance for max metrics --> Bigger values are better.
    df_normalized[max_metrics] = tolerance + (1 - 2 * tolerance) * (df[max_metrics] - min_values) / ranges

    # --- negative_metrics = ['train/actor_loss']
    if negative_metrics:
        min_values = df[negative_metrics].min()
        max_values = df[negative_metrics].max()
        ranges = max_values - min_values

        # Apply Min-Max Scaling with tolerance for negative metrics --> More negative values are better.
        df_normalized[negative_metrics] = tolerance + (1 - 2 * tolerance) * (max_values - df_normalized[negative_metrics]) / ranges
        
    # --- min_metrics = ['train/critic_loss', 'rollout/ep_len_mean', 'Avg time reach target']
    if min_metrics:
        min_values = df[min_metrics].min()
        max_values = df[min_metrics].max()
        ranges = max_values - min_values
        # Apply Min-Max Scaling with tolerance for min metrics --> Smaller values are better.
        df_normalized[min_metrics] = tolerance + (1 - 2 * tolerance) * ((df[min_metrics] - min_values) / ranges)
        # Get the inverse
        df_normalized[min_metrics] =1 - df_normalized[min_metrics] 

    # Apply tolerance to ensure values don't reach exactly 0 or 1
    df_normalized = df_normalized.apply(lambda x: np.clip(x, tolerance, 1 - tolerance))

    # Prepare the output: list of normalized data and names
    for id_, row in df.iterrows():
        action_time = row["Action time (s)"]
        names.append(f"T_{action_time:.2f}")
        data_list.append(df_normalized.loc[id_, labels].tolist())

    return data_list, names, labels


def plot_multiple_spider(data_list, labels, names, title='Models Comparison'):
    """
    Plots multiple spider charts on the same figure to compare different models.

    Args:
    - data_list (list of lists): A list of several lsit of metrics, one per model.
    - labels (list): List of labels for the axes (metrics).
    - names (list): List of names corresponding to each dataset (for the legend). They correspond to the action time in seconds.
    - title (str): The title of the chart.
    """
    # Vars number
    num_vars = len(labels)

    # Create angle for each category
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # Close th circle
    angles += angles[:1]

    # Create the figure
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100, subplot_kw=dict(polar=True))
    
    # Plot each data set
    for data, name in zip(data_list, names):
        data = data + data[:1]  # Assure that we are closing the circle
        ax.plot(angles, data, linewidth=2, linestyle='solid', label=name)
        ax.fill(angles, data, alpha=0.25)

    # Labels of the axis
    ax.set_yticklabels([])  # Remove labels from radial axis
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10, ha='center', rotation=60)
    ax.spines['polar'].set_visible(False)

    # Set the radial axis limits
    ax.set_ylim(0, 1) 

    # Add the leyend and title
    ax.legend(loc='upper center', bbox_to_anchor=(1.3, 1.1))  # Ajustar posición de la leyenda
    ax.set_title(title, size=16, color='black', y=1.1)

    # Show the plot
    plt.show()


def exponential_model(t, A, k, B):
    """
    Exponential model for modelling a first order system shifted in the y axis.
    A(1 - exp(-k * t)) + B
    """
    return A * (1 - np.exp(-k * t)) + B


def exponential_derivative(t, A, k, B):
    """
    Derivative of the exponential model.
    A * k * exp(-k * t)
    """
    return A * k * np.exp(-k * t)


def get_convergence_point_works (file_path, x_axis, convergence_threshold=0.01):
    # Load the CSV data
    df = pd.read_csv(file_path)
    
    # Get the x-axis values
    if x_axis =="Time":
        # Convert timestamps to relative time in hours
        df['Step timestamp'] = pd.to_datetime(df['Step timestamp'], format='%Y-%m-%d_%H-%M-%S')
        start_time = df['Step timestamp'].iloc[0]
        df['Relative time'] = (df['Step timestamp'] - start_time).dt.total_seconds() / 3600
        x_axis_values = df['Relative time'].values
    
    elif x_axis =="Step":
        x_axis_values = df['Step'].values
        # x_axis_values = x_axis_values/1000
        # x_axis_values = x_axis_values - x_axis_values[0]  # hace que empiece en 0

    # Get y-axis, which will be always the reward
    reward = df['rollout/ep_rew_mean'].values

    # Fit the exponential model
    initial_estimation = [np.max(reward), 0.1, np.min(reward)]  # Initial estimation for the parameters A,k,B

    popt, _ = curve_fit(exponential_model, x_axis_values, reward, p0=initial_estimation)
    A, k, B = popt  # Extract parameters of the exponential model
    
    # Compute the predicted values using the fitted model
    reward_fit = exponential_model(x_axis_values, A, k, B)
    
    # Compute the derivative of the exponential model
    reward_derivative = exponential_derivative(x_axis_values, A, k, B)
    print(reward_derivative)
    
    # We need to adapt the threshold for each x_axis.
    x_range = x_axis_values[-1] - x_axis_values[0]
    print(f"x_range: {x_range}")
    max_reward = np.max(reward)
    min_reward = np.min(reward)
    print(f"max_reward: {max_reward}")
    print(f"min_reward: {min_reward}")
    factor = 0.2
    convergence_threshold = factor * (np.max(reward) - np.min(reward)) / x_range   
    # convergence_threshold = factor * (np.max(reward) - np.min(reward)) / np.sqrt(x_range)
    # convergence_threshold = factor * (np.max(reward) - np.min(reward))
    print(f"convergence_threshold: {convergence_threshold}")

    # Find the point in the x-axis when the derivative crosses below the threshold or zero
    for i in range(1, len(reward_derivative)):
        if reward_derivative[i] < convergence_threshold and reward_derivative[i - 1] >= convergence_threshold:
            convergence_point = x_axis_values[i]
            print("break")
            break
    else:
        # TODO Check if this is a logical approach
        convergence_point = x_axis_values[-1]  # If no crossing found, return last x_axis value

    idx_convergence = np.argmin(np.abs(x_axis_values - convergence_point))
    print(idx_convergence)
    reward_at_convergence = reward[idx_convergence]
    return convergence_point, reward_fit, x_axis_values, reward, reward_at_convergence
    
### TODO Funciona igual que la anterior, comprobar si son equivalentes, y buscar una forma mejor de hacerlo
def get_convergence_point (file_path, x_axis, convergence_threshold=0.01):
    # Load the CSV data
    df = pd.read_csv(file_path)
    
    # Get the x-axis values
    if x_axis =="Time":
        # Convert timestamps to relative time in hours
        df['Step timestamp'] = pd.to_datetime(df['Step timestamp'], format='%Y-%m-%d_%H-%M-%S')
        start_time = df['Step timestamp'].iloc[0]
        df['Relative time'] = (df['Step timestamp'] - start_time).dt.total_seconds() / 3600
        x_axis_values = df['Relative time'].values
    
    elif x_axis =="Step":
        x_axis_values = df['Step'].values
        # x_axis_values = x_axis_values - x_axis_values[0]  # hace que empiece en 0

    # Get y-axis, which will be always the reward
    reward = df['rollout/ep_rew_mean'].values

    # Fit the exponential model
    initial_estimation = [np.max(reward), 0.1, np.min(reward)]  # Initial estimation for the parameters A,k,B

    popt, _ = curve_fit(exponential_model, x_axis_values, reward, p0=initial_estimation)
    A, k, B = popt  # Extract parameters of the exponential model
    
    # Compute the predicted values using the fitted model
    reward_fit = exponential_model(x_axis_values, A, k, B)
    
    # Compute the derivative of the exponential model
    reward_derivative = exponential_derivative(x_axis_values, A, k, B)
    print(reward_derivative)

    # Normalizar la derivada respecto a su valor máximo
    max_derivative = np.max(np.abs(reward_derivative))
    
    factor = 0.02
    # Umbral como porcentaje del valor máximo de la derivada
    convergence_threshold = factor * max_derivative

    print(f"convergence_threshold: {convergence_threshold}")

    # Find the point in the x-axis when the derivative crosses below the threshold or zero
    for i in range(1, len(reward_derivative)):
        if reward_derivative[i] < convergence_threshold:
            convergence_point = x_axis_values[i]
            print("break")
            break
    else:
        # TODO Check if this is a logical approach
        convergence_point = x_axis_values[-1]  # If no crossing found, return last x_axis value

    idx_convergence = np.argmin(np.abs(x_axis_values - convergence_point))
    print(idx_convergence)
    reward_at_convergence = reward[idx_convergence]
    return convergence_point, reward_fit, x_axis_values, reward, reward_at_convergence


def get_convergence_point_old(file_path, x_axis, convergence_threshold=0.01):
    df = pd.read_csv(file_path)
    
    if x_axis == "Time":
        df['Step timestamp'] = pd.to_datetime(df['Step timestamp'], format='%Y-%m-%d_%H-%M-%S')
        start_time = df['Step timestamp'].iloc[0]
        x_axis_values = (df['Step timestamp'] - start_time).dt.total_seconds() / 3600
    elif x_axis == "Step":
        x_axis_values = df['Step'].values
    
    x_axis_values = x_axis_values - x_axis_values[0]
    reward = df['rollout/ep_rew_mean'].values
    
    try:
        # Ajuste con límites para mayor estabilidad
        initial_estimation = [np.max(reward), 0.1, np.min(reward)] 
        bounds = ([0.5*np.max(reward), 0, 0], [2*np.max(reward), 10, np.max(reward)])
        popt, _ = curve_fit(exponential_model, x_axis_values, reward, p0=initial_estimation)
        A, k, B = popt
        reward_fit = exponential_model(x_axis_values, A, k, B)
        
        # Cálculo de target_reward más flexible
        target_reward = B + 0.8*(A - B)  # 90% del rango en lugar de 95%
        print(target_reward)
        
        # Encuentra todos los puntos que superan el target
        convergence_indices = np.where(reward_fit >= target_reward)[0]
        
        if len(convergence_indices) > 0:
            idx_convergence = convergence_indices[0]
            convergence_point = x_axis_values[idx_convergence]
            reward_at_convergence = reward[idx_convergence]
        else:
            # Fallback: usar el último punto si no se alcanza el target
            idx_convergence = len(x_axis_values) - 1
            convergence_point = x_axis_values[idx_convergence]
            reward_at_convergence = reward[idx_convergence]
            print("Warning: No se alcanzó el target de convergencia, usando el último punto")
            
    except Exception as e:
        print(f"Error en el ajuste exponencial: {e}")
        # Fallback: devolver valores por defecto
        idx_convergence = len(x_axis_values) - 1
        convergence_point = x_axis_values[idx_convergence]
        reward_fit = reward
        reward_at_convergence = reward[idx_convergence]
    
    return convergence_point, reward_fit, x_axis_values, reward, reward_at_convergence



# def plot_convergence_point (file_path, x_axis, convergence_threshold=0.01):
#     """
#     Finds the poitn at which the reward stabilizes (converges) based on a first-order fit.
    
#     Args:
#     - file_path (str): Path to the CSV file containing the reward data.
#     - convergence_threshold (float): Maximum slope value to consider the curve stabilized.
    
#     Returns:
#     - convergence_point (float): The estimated point in the x-axiswhen the reward stabilizes.
#     """
#     # Calculate the convergence time
#     convergence_point, reward_fit, x_axis_values, reward = get_convergence_point (file_path, x_axis, convergence_threshold)
    
#     # Plot results
#     plt.figure(figsize=(8, 5))
#     plt.plot(x_axis_values, reward, label='Original Data', marker='o', linestyle='')
#     plt.plot(x_axis_values, reward_fit, label='Exponential Fit', linestyle='--')
#     plt.axvline(convergence_point, color='r', linestyle=':', label=f'Convergence {x_axis}: {convergence_point:.2f}h')
#     if x_axis == "Time":
#         plt.xlabel('Time (hours)')
#         plt.title('Reward Convergence Analysis vs Time')
#     elif x_axis == "Step":
#         plt.xlabel('Step')
#         plt.title('Reward Convergence Analysis vs Step')
#     plt.ylabel('Reward')
#     plt.legend()
#     plt.grid()
#     plt.show()
    
#     return convergence_point


# def find_convergence_time(base_path, model_id, robot_name="turtleBot", window_fraction=1, slope_threshold=0.1):
#     """
#     Estimates the convergence time of the reward function by performing a linear fit
#     on the final portion of the training data.

#     Args:
#     - model_id (str): ID of the model to analyze.
#     - data_folder (str): Path to the folder containing the CSV files.
#     - threshold (float): Slope threshold to determine when the reward has stabilized.
#     - final_fraction (float): Fraction of the final data points used for linear fitting.

#     Returns:
#     - float: Estimated convergence time in hours, or None if convergence is not detected.
#     """
#     # Identify the CSV file matching the given model ID
#     file_pattern = f"{robot_name}_model_{model_id}_*.csv"
#     files = glob.glob(os.path.join(base_path, "robots", robot_name, "training_metrics", file_pattern))
    
#     if not files:
#         raise FileNotFoundError(f"No CSV file found for model ID {model_id}")
    
#     csv_file = files[0]  # Assume there is only one matching file
    
#     # Load the CSV data
#     df = pd.read_csv(csv_file)
    
#     # Convert timestamps to relative time (in hours)
#     # df["Step timestamp"] = pd.to_datetime(df["Step timestamp"])
#     # time_hours = (df["Step timestamp"] - df["Step timestamp"].iloc[0]).dt.total_seconds() / 3600
#     # rewards = df["rollout/ep_rew_mean"]

#     df['Step timestamp'] = pd.to_datetime(df['Step timestamp'], format='%Y-%m-%d_%H-%M-%S')
#     start_time = df['Step timestamp'].iloc[0]
#     df['Relative time'] = (df['Step timestamp'] - start_time).dt.total_seconds() / 3600  # Tiempo en horas    

#     time = df['Relative time'].values
#     reward = df['rollout/ep_rew_mean'].values
    
#     # Determine the subset of data for linear fitting
#     num_points = int(len(time) * window_fraction)
#     time_fit = time[-num_points:]
#     reward_fit = reward[-num_points:]
    
#     # Perform linear regression
#     slope, intercept, _, _, _ = linregress(time_fit, reward_fit)
    
#     # Estimate convergence time: Find the first time where the linear fit reaches stability
#     convergence_time = None
#     for t, r in zip(time, reward):
#         if abs(slope) <= slope_threshold:
#             convergence_time = t
#             break
    
#     if convergence_time is None:
#         print("Warning: Convergence time could not be determined within the dataset.")
#         return None
    
#     # Plot the reward evolution and linear fit
#     plt.figure(figsize=(8, 5))
#     plt.plot(time, reward, label='Reward', color='blue')
#     plt.plot(time_fit, intercept + slope * time_fit, label='Linear Fit', linestyle='dashed', color='red')
#     plt.axvline(convergence_time, color='green', linestyle='dotted', label=f'Convergence at {convergence_time:.2f}h')
#     plt.xlabel("Time (hours)")
#     plt.ylabel("Reward")
#     plt.title(f"Convergence Analysis for Model {model_id}")
#     plt.legend()
#     plt.show()
    
#     return convergence_time
    # """
    # Encuentra el tiempo de convergencia de la recompensa a partir de un archivo CSV.

    # Parámetros:
    # - model_id (str): ID del modelo a buscar.
    # - robot_name (str): Nombre del robot (por defecto, 'turtleBot').
    # - threshold (float): Valor mínimo de la derivada para considerar convergencia.

    # Retorna:
    # - None (muestra el gráfico con el ajuste y el tiempo de convergencia).
    # """

    # # Buscar el archivo CSV basado en el patrón
    # file_pattern = f"{robot_name}_model_{model_id}_*.csv"
    # files = glob.glob(os.path.join(base_path, "robots", robot_name, "training_metrics", file_pattern))
    
    # if not files:
    #     print(f"No se encontró archivo para el modelo {model_id}.")
    #     return
    
    # file_path = files[0]  # Tomar el primer archivo que coincida
    # print(f"Procesando archivo: {file_path}")

    # # Cargar el CSV
    # df = pd.read_csv(file_path)

    # # Convertir 'Step timestamp' a tiempo relativo en horas
    # df['Step timestamp'] = pd.to_datetime(df['Step timestamp'], format='%Y-%m-%d_%H-%M-%S')
    # start_time = df['Step timestamp'].iloc[0]
    # df['Relative time'] = (df['Step timestamp'] - start_time).dt.total_seconds() / 3600  # Tiempo en horas

    # # Obtener la recompensa media
    # time = df['Relative time'].values
    # reward = df['rollout/ep_rew_mean'].values

    # # Calcular derivada numérica
    # derivative = np.gradient(reward, time)

    # # Interpolación para suavizar la derivada
    # derivative_interp = interp1d(time, derivative, kind='cubic', fill_value="extrapolate")

    # # Buscar un punto donde la derivada esté cerca de 0
    # def target_function(t):
    #     return derivative_interp(t)
    
    # # Revisar si la derivada cambia de signo en el intervalo
    # if np.sign(target_function(time[0])) == np.sign(target_function(time[-1])):
    #     print("⚠️ Advertencia: La derivada no cruza cero en el intervalo.")
    #     print("Se usará el primer punto donde la derivada sea menor que el umbral.")

    #     # Buscar el primer índice donde la derivada es pequeña
    #     close_to_zero = np.abs(derivative) < threshold
    #     if np.any(close_to_zero):
    #         idx = np.argmax(close_to_zero)  # Primer índice donde la derivada es pequeña
    #         convergence_time = time[idx]
    #     else:
    #         print("No se encontró un tiempo de convergencia.")
    #         return
    # else:
    #     # Encontrar la raíz (cruce de cero)
    #     result = root_scalar(target_function, bracket=[time[0], time[-1]], method='brentq')

    #     if result.converged:
    #         convergence_time = result.root
    #     else:
    #         print("No se encontró un tiempo de convergencia.")
    #         return

    # print(f"Tiempo de convergencia estimado: {convergence_time:.2f} horas")

    # # Graficar los resultados
    # plt.figure(figsize=(8, 5))
    
    # # Gráfica original
    # plt.plot(time, reward, label="Recompensa", color='b', alpha=0.7)
    
    # # Marcar el tiempo de convergencia
    # plt.axvline(convergence_time, color='r', linestyle='--', label=f"Convergencia: {convergence_time:.2f}h")
    
    # # Etiquetas y título
    # plt.xlabel("Tiempo relativo (horas)")
    # plt.ylabel("Recompensa media")
    # plt.legend()
    # plt.title(f"Tiempo de convergencia del modelo {model_id}")
    # plt.grid()
    
    # # Mostrar el gráfico
    # plt.show()