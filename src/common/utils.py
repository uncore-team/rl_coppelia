import ast
from copy import deepcopy
import os
from collections import defaultdict
import csv
import curses
import datetime
import glob
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Dict, Iterable, List, Optional, Tuple, Union
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
import psutil
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from sklearn.covariance import MinCovDet
from tensorboard.backend.event_processing import event_accumulator
import threading
import select
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.evaluation import evaluate_policy
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

from PIL import Image
from scipy.spatial import cKDTree

from socketcomms.comms import BaseCommPoint

AGENT_SCRIPT_COPPELIA = "/Agent_Script"                     # Name of the agent script in CoppeliaSim scene
AGENT_SCRIPT_PYTHON = "coppelia_scripts/rl_script_copp.py"  # Script with the agent-specific functions: observations, rewards, step, reset, etc.

ROBOT_SCRIPT_COPPELIA = "/Robot_Script"                         # Name of the robot script in CoppeliaSim scene
ROBOT_SCRIPT_PYTHON = "coppelia_scripts/robot_script_copp.py"   # Script with the robot-specific functions: cmd_vel, draw path and graphs, ROScomm, etc.

OBSTACLES_GEN_SCRIPT_COPPELIA = "/ObstaclesGenerator"                   # Name of the obstacles generator script in CoppeliaSim scene
OBSTACLES_GEN_SCRIPT_PYTHON = "coppelia_scripts/generate_obstacles.py"  # Script with the obstacles generator functions.

# --- Path scene version, used for building timesteps map
AGENT_SCRIPT_PV_PYTHON = "coppelia_scripts/rl_script_pv_copp.py"  
ROBOT_SCRIPT_PV_PYTHON = "coppelia_scripts/robot_script_pv_copp.py"


# ------------------------------------------
# ------------------------------------------
# ------- Functions for managing logs ------
# ------------------------------------------
# ------------------------------------------

def initial_warnings(self):
    """
    Checks the provided arguments for missing values and sets default values if necessary.

    This function checks if certain required arguments are provided. If any of the arguments (`robot_name`, 
    `params_file`, `model_name`, `scene_path`) are missing or empty, the function prints the corresponding
    warning logs, and set some of them to their default values if neccesary.

    Args:
        args (Namespace): The command-line arguments passed to the script.

    Returns: None
    """
    
    if hasattr(self.args, "params_file") and not self.args.params_file:
        if self.args.command == 'train':
            self.args.params_file = os.path.join(self.base_path, "configs", f"params_default_file_{self.args.robot_name}.json")
            logging.warning(f"WARNING: '--params_file' was not specified, so the default file of the selected robot will be used: {self.args.params_file}.")
        elif self.args.command == 'test' or self.args.command == 'test_scene':
            logging.warning("WARNING: '--params_file' was not specified, so the json file used for training this model will be used.")

    if hasattr(self.args, "model_name") and not self.args.model_name:  
        logging.warning("WARNING: '--model_name' is required for testing functionality. The testing experiment will use the last saved model.")

    if hasattr(self.args, "scene_path") and not self.args.scene_path:
        logging.warning(f"WARNING: '--scene_path' was not specified, so default one will be used: <robot_name>_scene.ttt.")


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

    if save_files:
        if verbose==3:  # All through terminal and saved in log files
            log_handlers =[
                logging.StreamHandler(),  # Show through terminal
                rotating_handler          # Save in log files
            ]
        elif verbose==2:    # Just show the progress bar through terminal, but save everything in log files
            log_handlers =[rotating_handler]
        elif verbose ==1:   # Just show progress bar through terminal, and save just warning in log files
            # Set the log level for the rotating handler to WARNING
            rotating_handler.setLevel(logging.WARNING)  
            log_handlers =[rotating_handler]
        elif verbose ==0:   # Nothing in terminal, and just the errors will be saved in log files
            rotating_handler.setLevel(logging.ERROR) 
            log_handlers =[rotating_handler]
        else:
            log_handlers =[logging.StreamHandler()]
    else:   # Nothing will be saved in log files (actually it's a deprecated option)
        log_handlers =[logging.StreamHandler()]

    logging.basicConfig(
        level=log_level,  
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=log_handlers
    )


def logging_config_gui(log_level = logging.DEBUG):
    """
    Configures the logging system for the gui app.

    Args:
        log_level (optional): Logging level (default is logging.INFO).
    """
    logging.basicConfig(
        level=log_level,  
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers= [logging.StreamHandler()]
    )


# ------------------------------------------
# ------------------------------------------
# ------- Functions for communication ------
# ------------------------------------------
# ------------------------------------------


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


# -------------------------------------------------------------
# -------------------------------------------------------------
# ------ Functions for managing files, names and folders ------
# -------------------------------------------------------------
# -------------------------------------------------------------


def get_or_create_csv_path(base_model_name, metrics_folder, get_output_csv_func):
    """
    Get the path to an existing CSV file matching the model name, or create a new one.

    This function searches the specified metrics folder for a CSV file that matches the base model name.
    If one is found, it returns the path to the existing file. Otherwise, it generates a new file path
    using the provided CSV generation function.

    Args:
        base_model_name (str): The base name of the model (e.g., "turtleBot_model_15").
        metrics_folder (str): The folder where training/testing metric CSVs are stored.
        get_output_csv (Callable): Function used to generate a new CSV path if none is found.

    Returns:
        Tuple[str, bool]: A tuple containing:
            - experiment_csv_path (str): The path to the existing or newly created CSV.
            - csv_exists (bool): True if an existing CSV was found, False if a new one was created.
    """
    # Search for an existing CSV file matching the base model name
    pattern = os.path.join(metrics_folder, f"{base_model_name}*.csv")
    csv_matches = glob.glob(pattern)

    if csv_matches:
        experiment_csv_path = csv_matches[0]
        logging.info(f"Found existing CSV file: {experiment_csv_path}")
        csv_exists = True
    else:
        _, experiment_csv_path = get_output_csv(base_model_name, metrics_folder)
        logging.info(f"No existing CSV found. New file will be created at: {experiment_csv_path}")
        csv_exists = False

    return experiment_csv_path, csv_exists



def get_fixed_actimes(rl_copp_obj):
    """
    Given a list of model IDs, read their corresponding JSON parameter files
    and extract the value of 'params_env["fixed_actime"]'.

    Args:
        rl_copp_obj (object): An object containing the model IDs and the path to the parameter files.
        

    Returns:
        list of float: List of fixed_actime values for each model ID.

    Raises:
        FileNotFoundError: If a JSON file for a given model ID does not exist.
        KeyError: If 'params_env' or 'fixed_actime' is missing in the JSON.
    """
    actime_values = []

    for model_id in rl_copp_obj.args.model_ids:
        json_file = os.path.join(rl_copp_obj.paths["parameters_used"], f"params_default_file_{rl_copp_obj.args.robot_name}_model_{model_id}.json")
        
        if not os.path.isfile(json_file):
            raise FileNotFoundError(f"JSON file not found: {json_file}")
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        try:
            fixed_actime = data["params_env"]["fixed_actime"]
        except KeyError as e:
            raise KeyError(f"Missing key in JSON file {json_file}: {e}")
        
        actime_values.append(fixed_actime)

    return actime_values


def get_next_retrain_subfolder(log_dir):
    """
    Find the next available retrain subfolder name inside a given log directory.

    Args:
        log_dir (str): Path to the base TensorBoard log directory (e.g., tf_logs/turtleBot_tflogs_340).

    Returns:
        str: A subfolder name like "retrain_0", "retrain_1", etc.
    """
    existing = os.listdir(log_dir) if os.path.exists(log_dir) else []
    retrain_indices = [
        int(re.search(r"retrain_(\d+)", name).group(1))
        for name in existing
        if re.match(r"retrain_\d+", name)
    ]
    next_index = max(retrain_indices, default=-1) + 1
    return f"retrain_{next_index}"


def get_model_names_and_paths(rl_copp_obj): # TODO Fix that models are repeated as they are multiplied per the iterations
    '''
    For test_scene functionality
    '''
    model_names = {}
    model_paths = {}

    for model_id in rl_copp_obj.args.model_ids:
        model_id_str = str(model_id)
        model_name = f"{rl_copp_obj.args.robot_name}_model_{model_id_str}"
        model_dir = os.path.join(rl_copp_obj.paths["models"], model_name)
        model_path = os.path.join(model_dir, f"{model_name}_last")
        
        # If last model is not avaliable, then it will use one of the 'best' saved models' 
        if not os.path.exists(model_path + ".zip"):
            logging.warning(f"Last version of the model not found: {model_path}.zip")
            logging.info("Searching for other saved models for that experiment")
            pattern = os.path.join(model_dir, f"{model_name}_best*.zip")
            matches = glob.glob(pattern)
            if matches:
                model_path = os.path.splitext(matches[0])[0]
                logging.info(f"Using best model found: {model_path}.zip")
            else:
                logging.error(f"No best model found matching pattern: {pattern}")
                sys.exit()

        # Use last model by default
        else:
            logging.warning(f"Last version of the model has been found: {model_path}.zip")
            
        model_names[model_id_str] = model_name
        model_paths[model_id_str] = model_path

    return model_names, model_paths


def find_scene_csv_in_dir(folder_path):
    """
    Returns the path of the only CSV file in the given directory whose name starts with 'scene'.

    Args:
        folder_path (str): Path to the directory.

    Returns:
        str: Full path to the 'scene' CSV file found.

    Raises:
        ValueError: If no matching CSV is found or if multiple are found.
    """
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv') and f.startswith('scene')]

    if len(csv_files) == 0:
        raise ValueError(f"No 'scene*.csv' file found in {folder_path}")
    elif len(csv_files) > 1:
        raise ValueError(f"Multiple 'scene*.csv' files found in {folder_path}: {csv_files}")

    return os.path.join(folder_path, csv_files[0])


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


def get_file_index(args, tf_path, robot_name, retrain_flag=False):
    """
    Retrieves an index string for naming output files or logs based on the execution mode.

    In training mode (when `args` does not have `model_name` and is not a string), the function finds
    the next available index by scanning existing TensorBoard log directories matching the pattern 
    '{robot_name}_tflogs_<index>'.

    In testing mode (when `args` is a string or has `model_name`), the function extracts the model ID
    from the model name and appends a timestamp to create a unique identifier.

    Args:
        args (Union[argparse.Namespace, str]): Argument object or string representing the model name.
        tf_path (str): Path to the directory containing TensorBoard log folders.
        robot_name (str): Name of the robot, used in the naming convention of log folders.

    Returns:
        str: 
            - In training mode, the next available numerical index for new logs (e.g., "3").
            - In testing mode, a string composed of the model ID and a timestamp (e.g., "307_2025-05-07_14-20-01").
    """

    # TRAINING MODE: No model_name attribute and not a string → get next numerical index
    if not hasattr(args, "model_name") and not isinstance(args, str):
        tf_name = f"{robot_name}_tflogs"
        max_index = 0

        # Search for all existing log directories matching the pattern
        for path in glob.glob(os.path.join(tf_path, f"{glob.escape(tf_name)}_[0-9]*")):
            file_name = path.split(os.sep)[-1]  # Extract folder name
            ext = file_name.split("_")[-1]     # Get the numeric suffix

            # Check if the base name matches and suffix is a number
            if tf_name == "_".join(file_name.split("_")[:-1]) and ext.isdigit():
                if int(ext) > max_index:
                    max_index = int(ext)

        # Return next available index as string
        index = str(max_index + 1)

    # TESTING MODE: args is a string → parse model ID directly
    elif isinstance(args, str):
        model_name = args
        model_id = model_name.rsplit("_", 2)[-2]  # Extract second to last segment as ID
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        index = f"{model_id}_{timestamp}"

    # TESTING MODE v2: args has model_name → parse model ID using regex
    else:
        basename = os.path.basename(args.model_name)
        match = re.search(r'model_(\d+)', basename)
        model_id = match.group(1)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        index = f"{model_id}_{timestamp}"
    
    return index


def get_next_model_name(path, robot_name, next_index, callback_mode = False):
    """
    Generates a new model/callback filename based on the training stage.

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


def get_base_model_name (base_name):
    """
    Extracts the base model name by removing '_last' or '_best' and any following suffix.

    This function takes a model filename or identifier string and removes known suffixes
    such as '_last', '_best', or any extended suffix starting with those (e.g., 
    '_best_train_rw_197000.zip') to retrieve the original base model name.

    Args:
        base_name (str): The full model name or path containing optional suffixes.

    Returns:
        str: The cleaned base model name without '_last', '_best', or subsequent parts.
    """
    match = re.match(r"(.+?)_(last|best).*", base_name)
    if match:
        return match.group(1)
    return base_name 


def extract_model_id(path):
    """
    Extracts the numeric model ID from a model path like:
    'turtleBot_model_307/turtleBot_model_307_last' or
    'turtleBot_model_307/turtleBot_model_307_best_train_rw_197000'.

    Args:
        path (str): Path to the model file or directory.

    Returns:
        str: The numeric model ID as a string, e.g., '307', or None if not found.
    """
    base_name = os.path.basename(path)
    match = re.search(r'model_(\d+)', base_name)
    if match:
        return match.group(1)
    return None


def extract_robot_and_model_id(self, model_name: str) -> Tuple[str, int]:
    """Extract robot name and numeric model id from a model path.

    Example:
        'turtleBot_model_1070/turtleBot_model_1070_last.zip'
        -> ('turtleBot', 1070)

    Args:
        model_name: A path-like string pointing to a model, possibly with folders.

    Returns:
        (robot_name, model_id)

    Raises:
        ValueError: If the pattern '<robot>_model_<id>' cannot be found.
    """
    # Keep only the parent dir name: 'turtleBot_model_1070'
    # Path(...).parts is robust for both / and \ separators
    try:
        parent_dir = Path(model_name).parts[0]
    except Exception:
        parent_dir = model_name.split("/")[0].split("\\")[0]

    m = re.match(r"^(?P<robot>.+)_model_(?P<id>\d+)$", parent_dir)
    if not m:
        raise ValueError("model_name does not match '<robot>_model_<id>' in its parent directory part.")

    robot = m.group("robot")
    model_id = int(m.group("id"))
    return robot, model_id


def find_model_record_csv(self, base_path: str, robot_name: str, model_id: int) -> Optional[str]:
    """Find the per-model training CSV for a given robot and model id.

    Expected filename pattern:
        <robot>_model_<id>_train_YYYY-MM-DD_HH-MM-SS.csv

    Search path:
        robots/<robot_name>/training_metrics/

    Args:
        base_path: Project base path.
        robot_name: e.g., 'turtleBot'
        model_id: e.g., 1070

    Returns:
        Absolute path to the CSV if found, else None.

    Notes:
        If multiple files match (e.g., several training runs), we pick the most recent
        by parsing the timestamp from the filename; if parsing fails, we fall back to mtime.
    """
    import datetime as _dt

    metrics_dir = os.path.join(base_path, "robots", robot_name, "training_metrics")

    if not os.path.isdir(metrics_dir):
        return None

    pattern = os.path.join(metrics_dir, f"{robot_name}_model_{model_id}_train_*.csv")
    candidates = glob(pattern)

    if not candidates:
        return None

    def _ts_from_name(path: str) -> Optional[_dt.datetime]:
        # filename: <robot>_model_<id>_train_YYYY-MM-DD_HH-MM-SS.csv
        fname = os.path.basename(path)
        m = re.search(r"_train_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})\.csv$", fname)
        if not m:
            return None
        date_str = m.group(1)  # YYYY-MM-DD
        time_str = m.group(2)  # HH-MM-SS
        try:
            return _dt.datetime.strptime(date_str + " " + time_str, "%Y-%m-%d %H-%M-%S")
        except Exception:
            return None

    # Sort by parsed timestamp desc; fallback to mtime
    def _sort_key(path: str):
        ts = _ts_from_name(path)
        if ts is not None:
            return (0, ts)  # 0 = has timestamp, order by ts
        return (1, Path(path).stat().st_mtime)  # 1 = fallback, order by mtime

    candidates.sort(key=_sort_key, reverse=True)
    return candidates[0]


def remove_row_where_first_col_equals(self, csv_path: str, first_col_key: str) -> bool:
    """Remove rows from CSV where the first column equals 'first_col_key'. Keeps header if present.

    Args:
        csv_path: Absolute path to the CSV.
        first_col_key: Target string to match against the first column.

    Returns:
        True if at least one row was removed, False otherwise.
    """
    if not os.path.isfile(csv_path):
        print("no file")
        return False

    removed = False
    with open(csv_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        for idx, line in enumerate(lines):
            # preserve header if looks like header
            if idx == 0 and ("Exp_id" in line or "exp_id" in line or "EXP_ID" in line):
                f.write(line)
                continue
            first = line.split(",", 1)[0].strip()
            print(first)
            if first == first_col_key:

                removed = True
                continue
            f.write(line)

    return removed



def get_robot_paths(base_dir, robot_name, agent_logs = False):
    """
    Generate the paths for working with a robot, and create the needed folders if they don't exist

    Args:
        base_dir (str): base path where the subfolders will be craeted
        robot_name (str): name of the robot.
        agent_logs (bool, optional): For generating the script logs path needed for the agent.

    Returns:
        paths (dict): Dictionary with the given paths.
    """
    
    if not agent_logs:
        script_logs_name = "rl_logs"
    else:
        script_logs_name = "agent_logs"
        
    paths = {
        "models": os.path.join(base_dir, "robots", robot_name, "models"),
        "callbacks": os.path.join(base_dir, "robots",robot_name, "callbacks"),
        "tf_logs": os.path.join(base_dir, "robots",robot_name, "tf_logs"),
        "script_logs": os.path.join(base_dir, "robots",robot_name, "script_logs", script_logs_name),
        "testing_metrics": os.path.join(base_dir, "robots",robot_name, "testing_metrics"),
        "training_metrics": os.path.join(base_dir, "robots",robot_name, "training_metrics"),
        "parameters_used": os.path.join(base_dir, "robots",robot_name, "parameters_used"),
        "scene_configs": os.path.join(base_dir, "robots",robot_name, "scene_configs"),
        "configs": os.path.join(base_dir, "configs")
    }
    
    # Create the folders if they don't exist
    for path in paths.values():
        os.makedirs(path, exist_ok=True)

    return paths


def get_next_retrain_model_name(models_dir, base_model_name):
    """
    Finds the latest retrain version of a given base model in the models directory,
    and returns the name for the next retrain version by incrementing the index.

    Args:
        models_dir (str): Path to the directory containing model files.
        base_model_name (str): Base name of the model without extension 
                               (e.g. "turtleBot_model_306_last").

    Returns:
        str: Next model file name without extension 
             (e.g. "turtleBot_model_306_last_retrain2").
    """
    pattern = re.compile(rf"^{re.escape(base_model_name)}_retrain_(\d+)$")
    max_index = -1

    for filename in os.listdir(models_dir):
        name, _ = os.path.splitext(filename)
        match = pattern.match(name)
        if match:
            retrain_idx = int(match.group(1))
            max_index = max(max_index, retrain_idx)

    next_index = max_index + 1
    return f"{base_model_name}_retrain_{next_index}"


def _get_default_params ():
    """
    Private function for setting the different parameters to their default values, in case that reading the json fails.

    Args: None

    Return:
        dicts: Three dictionaries with the parameters of the environment, the training and the testing process.
    """
    params = {
        "params_scene": {
            "wheel_radius": 0.033,
            "distance_between_wheels": 0.16,
            "n_obstacles": 10,
            "diam_obstacles": 0.12,
            "height_obstacles": 0.25,
            "flag_grid": true,  # type: ignore
            "grid_visible": false,  # type: ignore
            "quads_x": 2,
            "quads_y": 2,
            "grid_rows_per_quad": 5,
            "grid_cols_per_quad": 5,
            "fixed_obs": true,  # type: ignore
            "outer_disk_rad": 0.1,
            "middle_disk_rad": 0.05,
            "inner_disk_rad": 0.006            
        },
        "params_env": {
            "fixed_actime": 0.75,
            "dist_thresh_finish_flag": 0.5,
            "reward_1": 0.25,
            "reward_2": 0.5,
            "reward_3": 1,
            "max_count": 400,
            "max_time": 40,
            "max_dist": 2,
            "finish_flag_penalty": -0.1,
            "overlimit_penalty": -0.5,
            "crash_penalty": -1,
            "max_crash_dist": 0.18,
            "max_crash_dist_critical": 0.18,
            "laser_observations": 8,
            "dim_action_space": 2,
            "action_names": [
                "linear",
                "angular"
            ],
            "action_bottom_limits": [ 
                -0.22,
                -2.84
            ],
            "action_upper_limits": [
                0.22,
                2.84
            ],
            "dim_observation_space": 10,
            "observation_names": [
                "laser_obs0",
                "laser_obs1",
                "laser_obs2",
                "laser_obs3",
                "laser_obs4",
                "laser_obs5",
                "laser_obs6",
                "laser_obs7",
                "distance",
                "angle"
            ],
            "observation_bottom_limits": [
                0.0,
                -3.141592653589793,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0
            ],
            "observation_upper_limits": [
                5.0,
                3.141592653589793,
                8.0,
                8.0,
                8.0,
                8.0,
                8.0,
                8.0,
                8.0,
                8.0
            ]
        },
        "params_train": {
            "sb3_algorithm": "SAC",
            "policy": "MlpPolicy",
            "total_timesteps": 1500000,
            "callback_frequency": 10000,
            "n_training_steps": 2048,
            "scene_name": "burgerBot_scene.ttt",
            "robot_handle": "/Burger",
            "robot_base_handle": "/Burger/base_link_visual",
            "laser_handle": "/Burger/Laser"
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
        params_scene (dict): Parameters for configuring the robot.
        params_env (dict): Parameters for configuring the environment.
        params_train (dict): Parameters for configuring the training process.
        params_test (dict): Parameters for configuring the testing process.
    """
    try:
        with open(file_path, 'r') as f:
            params_file = json.load(f)
            if params_file :
                params_scene = params_file["params_scene"]
                params_env = params_file["params_env"]
                params_train = params_file["params_train"]
                params_test = params_file["params_test"]
                logging.info(f"Configuration loaded successfully from {file_path}.")
                return params_scene, params_env, params_train, params_test
            else:
                logging.error("Failed to load configuration.")
                raise
            
    except Exception as e:
        logging.error(f"Error loading configuration file: {e}")
        raise
    

def smart_cast(v: str):
    v = v.strip()
    try: return json.loads(v)       # true/false/null, números, listas/dicts...
    except: pass
    try: return ast.literal_eval(v) # tuplas, etc.
    except: pass
    if v.lower() in ("true","false"):
        return v.lower() == "true"
    return v

_INDEX_RE = re.compile(r"([^\[\]]+)(?:\[(\d+)\])?")  # token o token[idx]


def deep_set_indexed(d: dict, dotted: str, value):
    parts = dotted.split(".")
    cur = d
    for i, part in enumerate(parts):
        m = _INDEX_RE.fullmatch(part)
        if not m: raise ValueError(f"Clave inválida: {part}")
        key, idx = m.group(1), m.group(2)
        last = (i == len(parts) - 1)

        if idx is None:
            # dict step
            if last:
                cur[key] = value
            else:
                if key not in cur or not isinstance(cur[key], (dict, list)):
                    cur[key] = {}
                if isinstance(cur[key], list):
                    raise TypeError(f"Se esperaba dict en '{key}' pero hay list.")
                cur = cur[key]
        else:
            # list step
            idx = int(idx)
            if key not in cur or not isinstance(cur[key], list):
                cur[key] = []
            lst = cur[key]
            if idx >= len(lst):
                lst.extend([None]*(idx - len(lst) + 1))
            if last:
                lst[idx] = value
            else:
                if lst[idx] is None:
                    lst[idx] = {}
                if not isinstance(lst[idx], dict):
                    raise TypeError(f"Se esperaba dict en '{key}[{idx}]'.")
                cur = lst[idx]


def parse_overrides_list(pairs):
    result = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"Override inválido (usa KEY=VALUE): {item}")
        key, val = item.split("=", 1)
        deep_set_indexed(result, key.strip(), smart_cast(val))
    return result


def deep_update(base: dict, patch: dict):
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def collect_unknown(patch: dict, ref: dict, prefix=""):
    unknown = []

    # Si el patch no es dict, no hay claves que recorrer
    if not isinstance(patch, dict):
        return unknown

    for k, v in patch.items():
        keypath = f"{prefix}.{k}" if prefix else k

        # Si ref no es dict o no tiene la clave, toda la rama es desconocida
        if not isinstance(ref, dict) or k not in ref:
            # aplanamos la rama para reportar todas las subclaves si v es dict
            if isinstance(v, dict):
                stack = [(keypath, v)]
                while stack:
                    base, sub = stack.pop()
                    unknown.append(base)
                    for kk, vv in sub.items():
                        subpath = f"{base}.{kk}"
                        if isinstance(vv, dict):
                            stack.append((subpath, vv))
                        else:
                            unknown.append(subpath)
            else:
                unknown.append(keypath)
            continue

        refk = ref[k]

        # Si v es dict y refk también dict, recursiona
        if isinstance(v, dict) and isinstance(refk, dict):
            unknown += collect_unknown(v, refk, keypath)
        # Si v es dict pero refk NO es dict, estás intentando profundizar donde no toca
        elif isinstance(v, dict) and not isinstance(refk, dict):
            # reporta toda la rama como desconocida
            stack = [(keypath, v)]
            while stack:
                base, sub = stack.pop()
                unknown.append(base)
                for kk, vv in sub.items():
                    subpath = f"{base}.{kk}"
                    if isinstance(vv, dict):
                        stack.append((subpath, vv))
                    else:
                        unknown.append(subpath)
        # Si v no es dict (scalar/list/etc.) y la clave existe, es OK → nada que hacer

    return unknown

    
def get_output_csv_deprecated(model_name, metrics_path, train_flag=True):   # TODO remove it's deprecated
    """
    Generate unique file names and paths for CSV files used to store model metrics.

    This function creates timestamped CSV file names based on the given model name.
    - In **training mode**, it generates one CSV file for storing training metrics.
    - In **inference (testing) mode**, it generates two CSV files: one for test metrics
      and another for additional data such as robot speeds.

    Args:
        model_name (str): Name of the model, used as a prefix in the CSV file names.
        metrics_path (str): Directory where the CSV files will be saved.
        train_flag (bool, optional): Indicates whether the script is running in 
            training mode (True) or inference/testing mode (False). Defaults to True.

    Returns:
        tuple:
            If ``train_flag`` is True:
                - output_csv_name (str): Name of the training CSV file.
                - output_csv_path (str): Full path to the training CSV file.
            
            If ``train_flag`` is False:
                - output_csv_name_1 (str): Name of the test metrics CSV file.
                - output_csv_name_2 (str): Name of the secondary (e.g., robot speeds) CSV file.
                - output_csv_path_1 (str): Full path to the test metrics CSV file.
                - output_csv_path_2 (str): Full path to the secondary CSV file.

    """
    # Get current timestamp so the metrics.csv file will have an unique name
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Get name and return the path to the csv file
    if train_flag:
        output_csv_name = f"{model_name}_train_{timestamp}.csv"
        output_csv_path = os.path.join(metrics_path, output_csv_name)
        return output_csv_name, output_csv_path
    else:
        output_csv_name_1 = f"{model_name}_test_{timestamp}.csv"
        output_csv_name_2 = f"{model_name}_otherdata_{timestamp}.csv"
        output_csv_path_1 = os.path.join(metrics_path, output_csv_name_1)
        output_csv_path_2 = os.path.join(metrics_path, output_csv_name_2)
        return output_csv_name_1, output_csv_name_2, output_csv_path_1, output_csv_path_2


def get_output_csv(model_name, metrics_path, file_type):
    """
    Generate unique file names and paths for CSV files used to store model metrics.

    Args:
        model_name (str): Name of the model, used as a prefix in the CSV file names.
        metrics_path (str): Directory where the CSV files will be saved.
        file_type (str): String that will be included in the file name (e.g. train, otherdata, test, etc.).

    Returns:
        tuple:
            
            output_csv_name (str): Name of the CSV file.
            output_csv_path (str): Full path to the CSV file.
    """
    # Get current timestamp so the metrics.csv file will have an unique name
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Get name and return the path to the csv file
    output_csv_name = f"{model_name}_{file_type}_{timestamp}.csv"
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
    new_duration = (end_time - start_time) / 3600
    new_end_time_str = datetime.datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')
    new_data = {
        "Exp_id": exp_name,
        "Start_time": datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S'),
        "End_time": new_end_time_str,
        "Duration": new_duration,
        **other_metrics
    }

    rows = []
    updated = False

    # Read existing data (if file exists)
    if os.path.exists(file_path):
        with open(file_path, mode="r", newline='') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames if reader.fieldnames else list(new_data.keys())

            for row in reader:
                if row["Exp_id"] == exp_name:
                    # Update this row
                    old_duration = float(row.get("Duration", 0.0))
                    new_data["Start_time"] = row["Start_time"]  # Keep original
                    new_data["Duration"] = old_duration + new_duration
                    rows.append({**row, **new_data})
                    updated = True
                else:
                    rows.append(row)
    else:
        headers = list(new_data.keys())

    if not updated:
        rows.append(new_data)

    # Write updated CSV
    with open(file_path, mode="w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    logging.info(f"Record file has been updated in {file_path}")


def copy_json_with_id(source_path, destination_dir, file_id):
    """
    Copies the given JSON file to a specified directory and appends a file_id to its name.

    Args:
        source_path (str): Path to the original JSON file.
        destination_dir (str): Directory where the file should be copied to.
        file_id (str): The file ID to append to the copied file's name.
    
    Returns:
        destination_path (str): The path to the copied JSON file with the new name.
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

    return destination_path


def get_data_from_training_csv(model_name, csv_path, column_header):
    """
    Searches for a row in the CSV file where the first column matches the given model name,
    and returns the value in the "column_header" column for that row. This is mainly used for
    getting the Algorithm used for training the model or for getting the final number of
    timesteps.

    Args:
        model_name (str): The model name to search for.
        csv_path (str): The path to the CSV file.
        column_header (str): Column name. Usually will be 'Algorithm' and 'Step'

    Returns:
        alg_name (str): The value from the "column_header" column corresponding to the row where 
                     the model name is found
    """
    # Just keep the name until the '_best' part
    if "_best" in model_name:
        model_name = model_name.split("_best")[0]

    with open(csv_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        # Get the key of the first column (assumed to contain the model names)
        first_col = reader.fieldnames[0]
        for row in reader:
            if row[first_col] == model_name:
                return row.get(column_header)

        logging.error(f"Error while getting data from csv")
        raise ValueError(f"There was an error while checking the {column_header} column for the {model_name} model. CSV file '{csv_path}'")


def get_params_file(paths, args, index_model_id = 0):
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
    if hasattr(args, "params_file"):
        if hasattr(args, "model_name"):
            if args.model_name is None:
                model_name, _ = get_last_model(models_path)
            else:
                model_name = args.model_name
        else:
            model_name = f"{args.robot_name}_model_{args.model_ids[index_model_id]}" #TODO Fix this, right now we are choosing the algorithm of the first model
            # (for test_scene functionality), so we cannot compare different models
    
    
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


def process_rl_exploitation_summary(summary_csv_path):
    """
    Processes a summary CSV of multiple RL exploitations, filters episodes with TimeSteps count == 1,
    and computes a cleaned summary CSV.

    Args:
        summary_csv_path (str): Path to the original summary CSV file.
    """
    # Load the original summary CSV
    summary_df = pd.read_csv(summary_csv_path)

    # Prepare list to store cleaned results
    cleaned_data = []

    # Define the output CSV name
    base_dir = os.path.dirname(summary_csv_path)
    print(f"Base directory for cleaned summary: {base_dir}")
    cleaned_summary_path = os.path.join(base_dir, "cleaned_summary.csv")

    for _, row in summary_df.iterrows():
        exp_id = row["Exp_id"]
        action_time = row["Action Time (s)"]

        if exp_id is None or pd.isna(exp_id):
            print("Warning: Exp_id is None or NaN. Skipping this row.")
            continue
        print(exp_id)

        folder_name = str(exp_id).split('last')[0] + 'last_testing'
        exp_path = os.path.join(base_dir, folder_name, exp_id)

        if not os.path.isfile(exp_path):
            print(f"Warning: File {exp_path} not found. Skipping.")
            continue

        # Load the exploitation CSV
        exp_df = pd.read_csv(exp_path)

        # Filter out episodes with TimeSteps count == 1
        filtered_df = exp_df[exp_df["TimeSteps count"] > 1]

        if filtered_df.empty:
            print(f"Warning: No valid episodes in {exp_id}. Skipping.")
            continue

        # Compute metrics
        avg_reward = filtered_df["Reward"].mean()
        avg_time = filtered_df["Time (s)"].mean()
        percentage_terminated = 100 * filtered_df["Terminated"].sum() / len(filtered_df)
        num_collisions = filtered_df["Crashes"].sum()
        collisions_percentage = 100 * num_collisions / len(filtered_df)
        zone_1_pct = 100 * (filtered_df["Target zone"] == 1).sum() / len(filtered_df)
        zone_2_pct = 100 * (filtered_df["Target zone"] == 2).sum() / len(filtered_df)
        zone_3_pct = 100 * (filtered_df["Target zone"] == 3).sum() / len(filtered_df)
        avg_distance = filtered_df["Distance traveled (m)"].mean()

        # Append cleaned result
        cleaned_data.append({
            "Exp_id": exp_id,
            "Action Time (s)": action_time,
            "Avg reward": round(avg_reward,3),
            "Avg time reach target": round(avg_time,3),
            "Percentage terminated": round(percentage_terminated,3),
            "Number of collisions": num_collisions,
            "Collisions percentage": round(collisions_percentage,3),
            "Target zone 1 (%)": round(zone_1_pct,3),
            "Target zone 2 (%)": round(zone_2_pct,3),
            "Target zone 3 (%)": round(zone_3_pct,3),
            "Avg episode distance (m)": round(avg_distance,3)
        })

    # Save cleaned summary to CSV
    cleaned_df = pd.DataFrame(cleaned_data)
    cleaned_df.to_csv(cleaned_summary_path, index=False)

    return cleaned_summary_path


def find_params_file(base_path, robot_name, experiment_id):
    """
    Finds the parameters file associated with a specific experiment ID for a given robot.
    Args:
        base_path (str): Base directory where robot data is stored.
        robot_name (str): Name of the robot.
        experiment_id (str): Experiment ID to search for.
    Returns:
        str: Path to the parameters file if found, otherwise None.
    """
    try:
        exp_id = f"{robot_name}_model_{experiment_id}"
        csv_path = os.path.join(base_path, "robots", robot_name, "training_metrics", "train_records.csv")

        df = pd.read_csv(csv_path).fillna("")
        row = df[df["Exp_id"] == exp_id]

        if not row.empty:
            params_file = row.iloc[0]["Params file"]
            return params_file.strip() if isinstance(params_file, str) else None
        else:
            logging.warning(f"Exp_id '{exp_id}' not found in {csv_path}")
            return None

    except Exception as e:
        logging.error(f"Error loading params file for {exp_id}: {e}")
        return None


# ---------------------------------------
# ---------------------------------------
# ------ Functions for CoppeliaSim ------
# ---------------------------------------
# ---------------------------------------


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


def stop_coppelia_simulation (self):
    """
    Check if Coppelia simulation is running and, in that case, it stops the simulation.

    Args:
        sim: CoppeliaSim object.
    """
    # Check simulation's state before stopping it
    if self.current_sim.getSimulationState() != self.current_sim.simulation_stopped:
        self.current_sim.stopSimulation()

        # Wait until the simulation is completely stopped
        while self.current_sim.getSimulationState() != self.current_sim.simulation_stopped:
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


def replace_std_variables(script_content, replacements):
    """
    Replaces standard variables in the script content with new values.

    Args:
        script_content (str): The original script content.
        replacements (dict): A dictionary where keys are variable names to be replaced
                             and values are the new values to replace them with.
    Returns:
        str: The updated script content with variables replaced.
    """
     # Update standard variables
    for var, new_value in replacements.items():
        if isinstance(new_value, str):
            script_content = re.sub(rf'{var}\s*=\s*["\'].*?["\']', f'{var} = "{new_value}"', script_content)
            script_content = re.sub(rf'{var}\s*=\s*None', f'{var} = "{new_value}"', script_content)
        else:
            script_content = re.sub(rf'{var}\s*=\s*\d+', f'{var} = {new_value}', script_content)
            script_content = re.sub(rf'{var}\s*=\s*None', f'{var} = {new_value}', script_content)

    return script_content


# Comments in English. Docstring uses Google style.

def _get_params_scene(rl_copp_obj) -> dict:
    """Extract `params_scene` from whatever container you use.

    Returns:
        dict: Scene parameters or empty dict if not present.
    """
    # Try known locations in your project structure
    if hasattr(rl_copp_obj, "params_scene") and isinstance(rl_copp_obj.params_scene, dict):
        return rl_copp_obj.params_scene
    if hasattr(rl_copp_obj, "params") and isinstance(rl_copp_obj.params, dict):
        if isinstance(rl_copp_obj.params.get("params_scene", {}), dict):
            return rl_copp_obj.params["params_scene"]
    # Fallback: nothing
    return {}


def _apply_scene_params(rl_copp_obj, script_path, fcn_name) -> None:
    """Push `params_scene` into the customization scripts of Coppelia.

    This function is used to update customziations cripts like these ones:
      - /Obs_Generator/script 
      - /Target/script  
      - /ExternalWall/script

    Args:
        rl_copp_obj: Manager with an active ZMQ `current_sim`.
        script_path (str): Path of the scriopt inside the CoppeliaSim scene (e.g. /Obs_Generator/script)
        fcn_name (str): Name of the function to be called within the script.
    """

    if not rl_copp_obj.params_scene:
        logging.warning("[Sim Scene] No params_scene provided; skipping.")
        return False

    try:
        script=rl_copp_obj.current_sim.getObject(script_path) 
        handle_script=rl_copp_obj.current_sim.getScript(rl_copp_obj.current_sim.scripttype_customization,script, fcn_name)

        rl_copp_obj.current_sim.callScriptFunction(fcn_name, handle_script, rl_copp_obj.params_scene)
        logging.info(f"[Sim Scene] Applied params_scene to {script_path} via {fcn_name} function.")
        return True
    except Exception as e:
        logging.warning(f"[Sim Scene] Could not call {fcn_name} in {script_path}: {e}")
        return False


def _build_replacements(
    rl_copp_obj,
    path_version: Optional[bool] = False
) -> Tuple[Dict, Dict, Dict]:
    """
    Build replacement dictionaries for agent, robot, and obstacle generator configuration.

    This function gathers, validates, and processes runtime attributes from the given
    `rl_copp_obj` (the main reinforcement learning manager for CoppeliaSim). It safely
    handles optional arguments, derives and normalizes variable names, expands model IDs
    according to the number of iterations per model and the number of targets found in
    the scene configuration file, and computes action times when applicable.

    It returns three dictionaries that can later be used to populate or substitute
    variables in configuration templates, simulator scripts, or environment setup files.

    Args:
        rl_copp_obj: The main Coppelia RL object containing runtime arguments (`args`),
            file paths (`paths`), scene and training parameters, and communication info.

    Returns:
        tuple:
            - replacements_agent (dict): Variables used by the RL agent, including
              model information, communication ports, and scene setup.
            - replacements_robot (dict): Robot geometry and alias parameters required
              for the robot script or simulator bindings.
            - replacements_obstacles_gen (dict): Parameters needed by the obstacle
              generator script (e.g., wheel spacing).

    Raises:
        SystemExit: If a required scene configuration CSV file cannot be found while
            expanding model IDs.
    """
    args = rl_copp_obj.args

    # Model name = basename or None
    if not hasattr(args, "model_name"):
        args.model_name = None
        model_name: Optional[str] = None
    else:
        try:
            model_name = os.path.basename(args.model_name) if args.model_name else None
        except Exception:
            model_name = None

    scene_to_load_folder = getattr(args, "scene_to_load_folder", None)
    obstacles_csv_folder = getattr(args, "obstacles_csv_folder", None)
    save_scene = getattr(args, "save_scene", None)
    save_traj = getattr(args, "save_traj", None)
    ip_address = getattr(args, "ip_address", None) or BaseCommPoint.get_ip()

    model_ids = getattr(args, "model_ids", None)
    action_times = None
    amp_model_ids = None

    # Model ids expansion if needed
    if model_ids is not None:
        # Get targets quantity
        if not scene_to_load_folder:
            logging.warning(
                "[UPDATING COPPELIA] 'scene_to_load_folder' not defined, so model_ids will not be expanded."
            )
        else:
            scene_configs_path = rl_copp_obj.paths["scene_configs"]
            scene_path = os.path.join(scene_configs_path, scene_to_load_folder)
            scene_path_csv = find_scene_csv_in_dir(scene_path)

            if not os.path.exists(scene_path_csv):
                logging.error(f"[UPDATING COPPELIA] CSV scene file not found: {scene_path_csv}")
                sys.exit(1)

            df = pd.read_csv(scene_path_csv)
            num_targets = int((df["type"] == "target").sum())

            # Repeat as many times as 'iters_per_model'
            iters_per_model = getattr(args, "iters_per_model", 1)
            amp_model_ids = [mid for mid in model_ids for _ in range(iters_per_model)]

            # If there are multiple targets, then multiply the obtained array
            if num_targets > 1:
                amp_model_ids *= num_targets

            # Update args and calculate action times
            rl_copp_obj.args.model_ids = amp_model_ids
            action_times = get_fixed_actimes(rl_copp_obj)

    # ----- Replacement dicts -----
    replacements_agent = {
        "robot_name": args.robot_name,
        "model_name": model_name,
        "model_ids": amp_model_ids,
        "base_path": rl_copp_obj.base_path,
        "comms_port": rl_copp_obj.free_comms_port,
        "ip_address": ip_address,
        "verbose": args.verbose,
        "scene_to_load_folder": scene_to_load_folder,
        "obstacles_csv_folder": obstacles_csv_folder,
        "save_scene": save_scene,
        "save_traj": save_traj,
        "testvar": rl_copp_obj.free_comms_port + 1,
        "action_times": action_times
    }

    if path_version:
        replacements_agent.update({
            "path_alias": args.path_alias,
            "trials_per_sample": args.trials_per_sample,
            "n_samples": args.n_samples,
            "n_extra_poses": args.n_extra_poses,
            "delta_deg": args.delta_deg,
            "place_obstacles_flag": args.place_obstacles_flag,
            "random_target_flag": args.random_target_flag,
            "map_png_path": args.map_png_path,
            "base_pos_samples": rl_copp_obj.base_pos_samples
        })

    replacements_robot = {
        "verbose": args.verbose,
        "distance_between_wheels": rl_copp_obj.params_scene["distance_between_wheels"],
        "wheel_radius": rl_copp_obj.params_scene["wheel_radius"],
        "robot_alias": rl_copp_obj.params_train["robot_handle"],
        "robot_base_alias": rl_copp_obj.params_train["robot_base_handle"],
        "laser_alias": rl_copp_obj.params_train["laser_handle"]
    }

    replacements_obstacles_gen = {
        "distance_between_wheels": rl_copp_obj.params_scene["distance_between_wheels"],
        "wheel_radius": rl_copp_obj.params_scene["wheel_radius"],
        "max_crash_dist_critical": rl_copp_obj.params_env["max_crash_dist_critical"],
        "outer_disk_rad": rl_copp_obj.params_scene["outer_disk_rad"]
    }

    return replacements_agent, replacements_robot, replacements_obstacles_gen


def _send_params_dict (rl_copp_obj, target_dict_name: str, script_content: str) -> str:
    """
    Insert a formatted parameter dictionary (params_env or params_scene) into a script template.

    This helper converts the specified parameter dictionary (either `params_env` or
    `params_scene` from `rl_copp_obj`) into a string formatted as valid Python code,
    and replaces the corresponding empty placeholder in the provided script content.

    Args:
        rl_copp_obj: The main RL CoppeliaSim manager object containing `params_env`
            and `params_scene` dictionaries.
        target_dict_name (str): Name of the target dictionary to inject into the script.
            Must be either "params_env" or "params_scene".
        script_content (str): The original script content containing the placeholder
            (e.g., `params_env = {}` or `params_scene = {}`).

    Returns:
        str: The updated script content with the formatted dictionary inserted.

    Raises:
        ValueError: If `target_dict_name` is not recognized.
    """
    params_str = "{"

    # Get the target dict from the taget dict name: params_scene, params_env, params_train, etc.
    target_dict = getattr(rl_copp_obj, target_dict_name)

    for key, value in target_dict.items():
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
    pattern = rf"{target_dict_name}\s*=\s*\{{\}}"
    script_content = re.sub(pattern, f"{target_dict_name} = {params_str}", script_content)

    return script_content


def update_and_copy_script_deprecated(rl_copp_obj):
    """
    Updates and copies both agent and robot scripts of the CoppeliaSim scene 
    with the content of the Python scripts specified in AGENT_SCRIPT_PYTHON, 
    ROBOT_SCRIPT_PYTHON and OBSTACLES_GEN_SCRIPT_PYTHON.

    This function loads both scripts from the specified paths, updates certain 
    variables in the scripts (such as robot name, base path, communication port,
    environment parameters, etc.), and sends the updated script content back to 
    the CoppeliaSim scene.

    Args:
        rl_copp_obj (RLCoppeliaManager): Object containing the simulation object, 
            base paths, parameters and command line arguments.

    Returns:
        bool: True if both scripts were successfully updated and copied to CoppeliaSim, 
              False otherwise.

    Raises:
        Exception: If an error occurs during the script update or copying process.
    """
    # Get script handles of CoppeliaSim scene
    try:
        agent_object = rl_copp_obj.current_sim.getObject(AGENT_SCRIPT_COPPELIA)
        agent_script_handle = rl_copp_obj.current_sim.getScript(1, agent_object)
    except Exception as e:
        raise ValueError(f"Error with agent script handle {AGENT_SCRIPT_COPPELIA}")

    robot_script_copp_path = f"{rl_copp_obj.params_train['robot_handle']}{ROBOT_SCRIPT_COPPELIA}"
    try:
        robot_object = rl_copp_obj.current_sim.getObject(robot_script_copp_path)
        robot_script_handle = rl_copp_obj.current_sim.getScript(1, robot_object)
    except Exception as e:
        raise ValueError(f"Error with robot script handle or script name {robot_script_copp_path}. Error: {e}")
    
    try:
        obstacles_gen_object = rl_copp_obj.current_sim.getObject(OBSTACLES_GEN_SCRIPT_COPPELIA)
        obstacles_gen_script_handle = rl_copp_obj.current_sim.getScript(1, obstacles_gen_object)
    except Exception as e:
        raise ValueError(f"Error with obstacles generator script handle {OBSTACLES_GEN_SCRIPT_COPPELIA}")


    # Get paths to the scripts
    agent_script_path = os.path.join(rl_copp_obj.base_path, "src", AGENT_SCRIPT_PYTHON)
    robot_script_path = os.path.join(rl_copp_obj.base_path, "src", ROBOT_SCRIPT_PYTHON)
    obstacles_gen_script_path = os.path.join(rl_copp_obj.base_path, "src", OBSTACLES_GEN_SCRIPT_PYTHON)

    logging.info(f"Copying content of {agent_script_path} inside the scene in {AGENT_SCRIPT_COPPELIA}")
    logging.info(f"Copying content of {robot_script_path} inside the scene in {ROBOT_SCRIPT_COPPELIA}")
    logging.info(f"Copying content of {obstacles_gen_script_path} inside the scene in {OBSTACLES_GEN_SCRIPT_COPPELIA}")


    # Read current content of the scripts
    with open(agent_script_path, "r") as file:
        agent_script_content = file.read()
    with open(robot_script_path, "r") as file:
        robot_script_content = file.read()
    with open(obstacles_gen_script_path, "r") as file:
        obstacles_gen_script_content = file.read()

    # Build dicts with variables to update
    replacements_agent, replacements_robot, replacements_obstacles_gen = _build_replacements(rl_copp_obj)

    # Update standard variables
    agent_script_content = replace_std_variables(agent_script_content, replacements_agent)
    robot_script_content = replace_std_variables(robot_script_content, replacements_robot)
    obstacles_gen_script_content = replace_std_variables(obstacles_gen_script_content, replacements_obstacles_gen)
    
    # Format the params dictionaries for updating the agent script
    agent_script_content = _send_params_dict(rl_copp_obj, "params_env", agent_script_content)
    agent_script_content = _send_params_dict(rl_copp_obj, "params_scene", agent_script_content)
   

    # Send updated scripts content to the scripts in CoppeliaSim
    rl_copp_obj.current_sim.setScriptText(agent_script_handle, agent_script_content)
    rl_copp_obj.current_sim.setScriptText(robot_script_handle, robot_script_content)
    rl_copp_obj.current_sim.setScriptText(obstacles_gen_script_handle, obstacles_gen_script_content)
    logging.info("[Sim Scene] Scripts updated successfully in CoppeliaSim.")

    # Override customization scripts 
    apply_params_errors = []
    apply_params_errors.append(_apply_scene_params(rl_copp_obj, "/Obs_Generator/script", "setObsParams"))
    apply_params_errors.append(_apply_scene_params(rl_copp_obj, "/Target/script", "setTargetParams"))
    apply_params_errors.append(_apply_scene_params(rl_copp_obj, "/ExternalWall/script", "setExternalWallParams"))
    
    if False in apply_params_errors:
        logging.error("[Sim Scene] Some customization scripts could not be updated with the specified params.")
    else:
        logging.info("[Sim Scene] Customization scripts of the CoppeliaSim scene have been overrided with the specified params in json file.")

    return True


def update_and_copy_script(rl_copp_obj, path_version = False):   # TODO Review Docstring (it's copypasted from twin function), and consider the option of splitting the original function
    """
    Updates and copies both agent and robot scripts of the CoppeliaSim scene 
    with the content of the Python scripts specified in AGENT_SCRIPT_PYTHON, 
    ROBOT_SCRIPT_PYTHON and OBSTACLES_GEN_SCRIPT_PYTHON.

    This function loads both scripts from the specified paths, updates certain 
    variables in the scripts (such as robot name, base path, communication port,
    environment parameters, etc.), and sends the updated script content back to 
    the CoppeliaSim scene.

    Args:
        rl_copp_obj (RLCoppeliaManager): Object containing the simulation object, 
            base paths, parameters and command line arguments.

    Returns:
        bool: True if both scripts were successfully updated and copied to CoppeliaSim, 
              False otherwise.

    Raises:
        Exception: If an error occurs during the script update or copying process.
    """
    # ----- Get script handles of CoppeliaSim scene
    try:
        agent_object = rl_copp_obj.current_sim.getObject(AGENT_SCRIPT_COPPELIA)
        agent_script_handle = rl_copp_obj.current_sim.getScript(1, agent_object)
    except Exception as e:
        raise ValueError(f"Error with agent script handle {AGENT_SCRIPT_COPPELIA}")

    robot_script_copp_path = f"{rl_copp_obj.params_train['robot_handle']}{ROBOT_SCRIPT_COPPELIA}"
    try:
        robot_object = rl_copp_obj.current_sim.getObject(robot_script_copp_path)
        robot_script_handle = rl_copp_obj.current_sim.getScript(1, robot_object)
    except Exception as e:
        raise ValueError(f"Error with robot script handle or script name {robot_script_copp_path}. Error: {e}")
    
    try:
        obstacles_gen_object = rl_copp_obj.current_sim.getObject(OBSTACLES_GEN_SCRIPT_COPPELIA)
        obstacles_gen_script_handle = rl_copp_obj.current_sim.getScript(1, obstacles_gen_object)
    except Exception as e:
        raise ValueError(f"Error with obstacles generator script handle {OBSTACLES_GEN_SCRIPT_COPPELIA}")


    # ----- Get paths to the scripts
    # Normal scene version
    if not path_version:
        agent_script_path = os.path.join(rl_copp_obj.base_path, "src", AGENT_SCRIPT_PYTHON)
        robot_script_path = os.path.join(rl_copp_obj.base_path, "src", ROBOT_SCRIPT_PYTHON)
    else:
        agent_script_path = os.path.join(rl_copp_obj.base_path, "src", AGENT_SCRIPT_PV_PYTHON)
        robot_script_path = os.path.join(rl_copp_obj.base_path, "src", ROBOT_SCRIPT_PV_PYTHON)

    obstacles_gen_script_path = os.path.join(rl_copp_obj.base_path, "src", OBSTACLES_GEN_SCRIPT_PYTHON)

    logging.info(f"Copying content of {agent_script_path} inside the scene in {AGENT_SCRIPT_COPPELIA}")
    logging.info(f"Copying content of {robot_script_path} inside the scene in {ROBOT_SCRIPT_COPPELIA}")
    logging.info(f"Copying content of {obstacles_gen_script_path} inside the scene in {OBSTACLES_GEN_SCRIPT_COPPELIA}")


    # ----- Read current content of the scripts
    with open(agent_script_path, "r") as file:
        agent_script_content = file.read()
    with open(robot_script_path, "r") as file:
        robot_script_content = file.read()
    with open(obstacles_gen_script_path, "r") as file:
        obstacles_gen_script_content = file.read()

    # ----- Build dicts with variables to update
    replacements_agent, replacements_robot, replacements_obstacles_gen = _build_replacements(rl_copp_obj, path_version)

    # Update standard variables
    agent_script_content = replace_std_variables(agent_script_content, replacements_agent)
    robot_script_content = replace_std_variables(robot_script_content, replacements_robot)
    obstacles_gen_script_content = replace_std_variables(obstacles_gen_script_content, replacements_obstacles_gen)
    
    # Format the params dictionaries for updating the agent script
    agent_script_content = _send_params_dict(rl_copp_obj, "params_env", agent_script_content)
    agent_script_content = _send_params_dict(rl_copp_obj, "params_scene", agent_script_content)
   

    # Send updated scripts content to the scripts in CoppeliaSim
    rl_copp_obj.current_sim.setScriptText(agent_script_handle, agent_script_content)
    rl_copp_obj.current_sim.setScriptText(robot_script_handle, robot_script_content)
    rl_copp_obj.current_sim.setScriptText(obstacles_gen_script_handle, obstacles_gen_script_content)
    logging.info("[Sim Scene] Scripts updated successfully in CoppeliaSim.")

    # Override customization scripts 
    apply_params_errors = []
    apply_params_errors.append(_apply_scene_params(rl_copp_obj, "/Obs_Generator/script", "setObsParams"))
    apply_params_errors.append(_apply_scene_params(rl_copp_obj, "/Target/script", "setTargetParams"))
    apply_params_errors.append(_apply_scene_params(rl_copp_obj, "/ExternalWall/script", "setExternalWallParams"))
    
    if False in apply_params_errors:
        logging.error("[Sim Scene] Some customization scripts could not be updated with the specified params.")
    else:
        logging.info("[Sim Scene] Customization scripts of the CoppeliaSim scene have been overrided with the specified params in json file.")

    return True


def create_discs_under_target(rl_copp_obj):     # TODO Remove, it's deprecated
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
    target_handle = rl_copp_obj.current_sim.getObject("/Target")

    # Remove any existing child discs
    child_objects = rl_copp_obj.current_sim.getObjectsInTree(target_handle, rl_copp_obj.current_sim.handle_all, 1)  # Get direct children
    for child in child_objects:
        obj_type = rl_copp_obj.current_sim.getObjectType(child)
        if obj_type == rl_copp_obj.current_sim.object_shape_type:  # Ensure it's a shape before deleting
            rl_copp_obj.current_sim.removeObject(child)

    # Disc properties: name, color (R, G, B), relative Z position
    disc_properties = [
        ("Target_disc_1", [0, 0, 1], 0.002, rl_copp_obj.params_scene["outer_disk_rad"]),  # Blue
        ("Target_disc_2", [1, 0, 0], 0.004, rl_copp_obj.params_scene["middle_disk_rad"]),  # Red
        ("Target_disc_3", [1, 1, 0], 0.006, rl_copp_obj.params_scene["inner_disk_rad"])   # Yellow
    ]

    disc_handles = []

    for name, color, z_offset, radius in disc_properties:
        # Create a disc with minimal thickness
        disc_handle = rl_copp_obj.current_sim.createPrimitiveShape(rl_copp_obj.current_sim.primitiveshape_disc, [radius * 2, radius * 2, 0.01], 0)

        # Set the disc's alias (name in the scene)
        rl_copp_obj.current_sim.setObjectAlias(disc_handle, name)

        # Set the disc as a child of the target object
        rl_copp_obj.current_sim.setObjectParent(disc_handle, target_handle, True)

        # Adjust the relative position (lifting it in the Z axis)
        rl_copp_obj.current_sim.setObjectPosition(disc_handle, target_handle, [0, 0, z_offset])

        # Set the color (ambient diffuse component)
        rl_copp_obj.current_sim.setShapeColor(disc_handle, None, rl_copp_obj.current_sim.colorcomponent_ambient_diffuse, color)

        # Store the handle in the list
        disc_handles.append(disc_handle)

    return disc_handles  # Return the handles of the created discs


def start_coppelia_and_simulation(rl_copp_obj, process_name:str, path_version:Optional[bool]=False):
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
    if rl_copp_obj.args.scene_path is None:

        # Get scene name from params file json
        scene_name = rl_copp_obj.params_train.get("scene_name", None)
        if not scene_name:
            raise ValueError("No 'scene_name' found in params_train.")
        
        # Build scene path
        scene_path = os.path.join(rl_copp_obj.base_path, "scenes", scene_name)

        # Check file existence
        if not os.path.isfile(scene_path):
            raise FileNotFoundError(f"Scene file not found: {scene_path}")
        
        rl_copp_obj.args.scene_path = scene_path

    # Verify if CoppeliaSim is running
    # TODO Check that when we open several instances of CoppeliaSim with the GUI, only
    # one will be preserved if the screen powers off automatically for saving energy.
    # So please use the no_gui mode for the moment if you are leaving the PC.
    if rl_copp_obj.args.dis_parallel_mode:
        if not is_coppelia_running():
            logging.info("Initiating CoppeliaSim...")
            if rl_copp_obj.args.no_gui:
                process = subprocess.Popen([
                    "gnome-terminal", 
                    f"--title={rl_copp_obj.terminal_pid}",
                    "--",
                    coppelia_exe, "-h"])
            else:
                process = subprocess.Popen(["gnome-terminal", "--",coppelia_exe])
        else:
            logging.info("CoppeliaSim was already running")
    else:
        logging.info("Initiating a new CoppeliaSim instance...")
        zmq_port = find_next_free_port(zmq_port)    
        ws_port = find_next_free_port(ws_port)

        # Save the PID of the terminal
        if hasattr(rl_copp_obj.args, "timestamp") and rl_copp_obj.args.timestamp is not None:
            timestamp = rl_copp_obj.args.timestamp  # Obtained from GUI
        else:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            
        rl_copp_obj.terminal_pid = f"{process_name} - {timestamp}"

        if rl_copp_obj.args.no_gui:
            process = subprocess.Popen([
                "gnome-terminal", 
                f"--title={rl_copp_obj.terminal_pid}",  # Title for identifying it
                "--", 
                "bash", "-c", 
                f"{coppelia_exe} -h -GzmqRemoteApi.rpcPort={zmq_port} -GwsRemoteApi.port={ws_port}; exec bash"
            ])
        
        else:
            # process = subprocess.Popen(["gnome-terminal", "--",coppelia_exe, f"-GzmqRemoteApi.rpcPort={zmq_port}", f"-GwsRemoteApi.port={ws_port}"])
            process = subprocess.Popen([
                "gnome-terminal", 
                f"--title={rl_copp_obj.terminal_pid}",
                "--", 
                "bash", "-c", 
                f"{coppelia_exe} -GzmqRemoteApi.rpcPort={zmq_port} -GwsRemoteApi.port={ws_port}; exec bash"
            ])
        

    # Get the id of the new process.
    rl_copp_obj.current_coppelia_pid = get_new_coppelia_pid(rl_copp_obj.before_pids)
            
    # Wait for CoppeliaSim connection
    try:
        logging.info("Waiting for connection with CoppeliaSim...")
        rl_copp_obj.client = RemoteAPIClient(port=zmq_port)
        rl_copp_obj.current_sim = rl_copp_obj.client.getObject('sim')
    except Exception as e:
        logging.error(f"It was not possible to connect with CoppeliaSim: {e}")
        if process:
            process.terminate()
        return False

    logging.info("Connection established with CoppeliaSim")

    # Check if scene is loaded
    if is_scene_loaded(rl_copp_obj.current_sim, rl_copp_obj.args.scene_path):
        logging.info("Scene is already loaded, simulation will be stopped in case that it's running...")
        stop_coppelia_simulation(rl_copp_obj)
    else:
        logging.info(f"Loading scene: {rl_copp_obj.args.scene_path}")
        try:
            rl_copp_obj.current_sim.loadScene(rl_copp_obj.args.scene_path)
        except:
            # Build scene path
            scene_path = os.path.join(rl_copp_obj.base_path, "scenes", rl_copp_obj.args.scene_path)

            # Check file existence
            if not os.path.isfile(scene_path):
                raise FileNotFoundError(f"Scene file not found: {scene_path}")
            rl_copp_obj.current_sim.loadScene(scene_path)
        logging.info("Scene loaded successfully.")

    # Update code inside Coppelia's scene
    update_and_copy_script(rl_copp_obj, path_version)

    # Start the simulation
    rl_copp_obj.current_sim.startSimulation()
    logging.info("Simulation started")


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


def close_coppelia_sim(current_pid, terminal_pid):
    logging.info(f"Closing CoppeliaSim processes with PIDs: {current_pid} and terminal {terminal_pid}")
    # Close the terminal
    try:
        # result = subprocess.run(['pkill', '-f', f'gnome-terminal.*{terminal_pid}'], check=False)
        result = subprocess.run(['wmctrl', '-c', terminal_pid], check=False)


        if result.returncode == 0:
            logging.info(f"Terminal {terminal_pid} closed successfully.")
        else:
            logging.warning(f"No process found with title {terminal_pid}")
    except:
        if current_pid:
            try:
                for pid in current_pid:
                    time.sleep(0.5)
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


# ---------------------------------------------
# ---------------------------------------------
# ------ Other main supporting functions ------
# ---------------------------------------------
# ---------------------------------------------


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

    # Get current opened processes in the PC so later we can know which ones are the Coppelia new ones.
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
        model_folder = args.robot_name + "_model_" + str(model_id)
        model_name = model_folder + "_last"
        cmd.extend(["--model_name", f"{model_folder}/{model_name}"])

    # Add the flag to suppress the GUI if specified
    if no_gui:
        cmd.append("--no_gui")

    # Add the disable parallel mode flag if you want to run the training sequentially
    if args.dis_parallel_mode:
        cmd.append("--dis_parallel_mode")

    # Add the robot name
    if args.robot_name:
        cmd.extend(["--robot_name", args.robot_name])

    if args.timestamp:
        cmd.extend(["--timestamp", args.timestamp])

    # Add the iterations
    if hasattr(args, "iterations") and args.iterations is not None:
        cmd.extend(["--iterations", str(args.iterations)])

    # Add the verbose mode
    if args.verbose:
        cmd.extend(["--verbose", str(args.verbose)])

    logging.info(f"CMD to be executed: {cmd}")

    try:
        # Run the command as a subprocess and capture the output
        process = subprocess.Popen(cmd, stderr=subprocess.STDOUT, text=True)        
        time.sleep(2)

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
                    time.sleep(3)
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
        

# ------------------------------------
# ------------------------------------
# ------ Functions for training ------
# ------------------------------------
# ------------------------------------


class StopTrainingOnKeypress(BaseCallback): # TODO Check: confirmation message appears two times
    """
    Callback that pauses training when a specific key is pressed and asks for confirmation
    to either stop or resume the training.

    When the designated key (default "F") is pressed, training is paused and the user is prompted 
    to confirm whether to stop training. If the user confirms with 'Y', training stops; if 'N', 
    training resumes. During the pause, the callback blocks training steps until a decision is made.
    """

    def __init__(self, key="F", verbose=1):
        super().__init__(verbose)
        self.key = key
        self.stop_training = False
        self.pause_event = threading.Event()
        self.pause_event.set()  # Training is not paused initially.
        self.confirmation_in_progress = False
        self.listener_thread = threading.Thread(target=self._listen_for_key, daemon=True)
        self.listener_thread.start()

    def _listen_for_key(self):
        """Thread that listens for a key press to trigger pause and confirmation."""
        print(f"Press '{self.key}' to pause training and request stop confirmation...")
        while not self.stop_training:
            if select.select([sys.stdin], [], [], 0.1)[0]:
                user_input = sys.stdin.read(1)
                if user_input.strip() == self.key and not self.confirmation_in_progress:
                    self.confirmation_in_progress = True
                    self.pause_event.clear()  # Pause training.
                    self._ask_for_confirmation()
                    self.confirmation_in_progress = False

    def _ask_for_confirmation(self):
        """
        Prompts the user for confirmation to stop training.
        The training is paused until the user inputs a valid answer.
        """
        print("\nTraining paused. Waiting for confirmation...")
        while True:
            if self.confirmation_in_progress:
                user_input = input("Do you really want to stop training? (Y/N): ").strip().upper()
                if user_input == 'Y':
                    self.stop_training = True
                    print("Training will be stopped.")
                    self.pause_event.set()  # Allow the training loop to exit.
                    break
                elif user_input == 'N':
                    print("Resuming training...")
                    self.pause_event.set()  # Resume training.
                    break
                else:
                    print("Invalid input. Please press 'Y' or 'N'.")

    def _on_step(self) -> bool:
        """
        Called at each training step. If training is paused, this method blocks until the user resumes or stops training.
        
        Returns:
            bool: False if stop_training is True, otherwise True to continue training.
        """
        while not self.pause_event.is_set() and not self.stop_training:
            time.sleep(0.1)
        return not self.stop_training


def parse_tensorboard_logs(
    log_dir,
    output_csv,
    metrics=[
        "train/loss", "train/actor_loss", "train/critic_loss", "train/entropy_loss",
        "train/value_loss", "train/approx_kl", "train/clip_fraction", "train/explained_variance",
        "train/ent_coef", "train/ent_coef_loss", "train/learning_rate",
        "rollout/ep_len_mean", "rollout/ep_rew_mean", "custom/sim_time", "custom/episodes"
    ]
):
    """
    Parse TensorBoard logs from a directory and its subdirectories, saving the selected metrics to a CSV file.

    The function searches recursively for TensorBoard event files and processes them in temporal order. All metric
    data is combined and written to the same output CSV. The last row returned corresponds to the final training entry.

    Args:
        log_dir (str): Root directory containing TensorBoard logs (including subfolders like retrain_0/).
        output_csv (str): Path to the CSV file to create or append to.
        metrics (List[str], optional): List of scalar tags to extract. Defaults to common training tags.

    Returns:
        Tuple[List[dict], dict]: A list of all rows written, and the last row dictionary.
    """

    def find_event_dirs(base_dir):
        """Recursively find directories containing TensorBoard event files."""
        event_dirs = []
        for root, _, files in os.walk(base_dir):
            if any("tfevents" in f for f in files):
                event_dirs.append(root)
        return sorted(event_dirs)  # Sorted alphabetically (and by depth)

    all_rows = []
    last_row_dict = {}

    event_dirs = find_event_dirs(log_dir)
    if not event_dirs:
        logging.error(f"No TensorBoard event files found under {log_dir}")
        return [], {}

    for subdir in event_dirs:
        try:
            ea = event_accumulator.EventAccumulator(subdir)
            ea.Reload()
        except Exception as e:
            logging.warning(f"Skipping {subdir}: failed to load events. Error: {e}")
            continue

        available_tags = ea.Tags().get("scalars", [])
        metrics_present = [m for m in metrics if m in available_tags]
        if not metrics_present:
            logging.info(f"No relevant metrics found in {subdir}. Skipping.")
            continue

        events_dict = {m: ea.Scalars(m) for m in metrics_present}
        n_events = min(len(v) for v in events_dict.values())

        for i in range(n_events):
            step = events_dict[metrics_present[0]][i].step
            wall_time = events_dict[metrics_present[0]][i].wall_time
            formatted_time = datetime.datetime.fromtimestamp(wall_time).strftime("%Y-%m-%d_%H-%M-%S")

            row = {"Step": step, "Step timestamp": formatted_time}
            for m in metrics:
                row[m] = events_dict[m][i].value if m in events_dict else ''
            all_rows.append((wall_time, row))  # Store with timestamp for later sorting

    if not all_rows:
        logging.error("No valid event data found in any log directory.")
        sorted_rows = []
        last_row_dict = []

    else:
        # Sort rows by wall_time to ensure chronological order
        all_rows.sort(key=lambda x: x[0])
        sorted_rows = [r for _, r in all_rows]
        last_row_dict = sorted_rows[-1]

    # Write to CSV
    headers = ["Step", "Step timestamp"] + metrics
    with open(output_csv, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(sorted_rows)

    logging.info(f"Combined TensorBoard metrics written to {output_csv}")

    return sorted_rows, last_row_dict


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving the model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    Args:
        check_freq (int): Frequency (in training steps) at which the model should be evaluated during training.
        log_dir (str): Path to the directory where the model will be saved. Must contain the file generated by the ``Monitor`` wrapper.
        verbose (int): Verbosity level.
    """

    def __init__(self, check_freq, save_path, log_dir, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        model_name = os.path.basename(save_path)
        self.save_path = os.path.join(save_path, f"{model_name}_best_train_rw")
        self.best_mean_reward = -np.inf

        # Pattern to locate old best model files in the save directory
        self.pattern = os.path.join(save_path, "*_best_train_rw_*.zip")

    def _on_step(self) -> bool:
        if self.n_calls>15000 and self.n_calls % self.check_freq == 0:  # default: 10K
            logging.info("Evaluating model")

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                y_float = []
                # Mean training reward over the last 50 episodes
                for val in y:
                    try:
                        y_float.append(float(val))
                    except ValueError:
                        logging.error(f"Wrong value removed: {val}")
                
                if y_float:
                    mean_reward = np.mean(y_float[-50:])
                else:
                    mean_reward = -99

                if self.verbose > 0:
                    logging.info(
                        "Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(
                            self.best_mean_reward, mean_reward
                        )
                    )

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward

                    # Remove any previous best models matching the pattern
                    for file_path in glob.glob(self.pattern):
                        try:
                            os.remove(file_path)
                            logging.info(f"Removed previous best (train) model: {file_path}")
                        except OSError as e:
                            logging.error(f"Error: It was not possible to remove the file {file_path}: {e}")

                    # Saving best model
                    if self.verbose > 0:
                        logging.info(f"Saving new best model to {self.save_path}_{self.num_timesteps}")
                    self.model.save(f"{self.save_path}_{self.num_timesteps}")

        return True


class CustomEvalCallback(EvalCallback):
    """Custom evaluation callback for Stable-Baselines3 with CoppeliaSim integration.

    This callback evaluates the policy every `eval_freq` steps using a separate 
    evaluation environment managed by a custom RL manager. It waits for an episode 
    to finish before performing the evaluation to ensure consistency in environments 
    like CoppeliaSim, where simulation time continues independently.

    Additionally, before saving a new best model (based on evaluation reward), it 
    deletes any previous best model files matching a specific naming pattern 
    (`*_best_test_rw_*.zip`) in the save directory.

    Attributes:
        rl_manager (RLManager): Custom manager containing the evaluation environment.
        steps_since_eval (int): Counter to track steps since the last evaluation.
        pattern (str): Glob pattern to identify old best model files to delete.
    """

    def __init__(self, *args, rl_manager, **kwargs):
        """Initializes the callback and configures model file pattern.

        Args:
            rl_manager (RLCoppeliaManager, optional): Custom manager with the evaluation environment.
            *args: Positional arguments for the EvalCallback.
            **kwargs: Keyword arguments for the EvalCallback.
        """
        super().__init__(*args, **kwargs)
        self.rl_manager = rl_manager
        self.steps_since_eval = 0

        # Pattern to locate old best model files in the save directory
        self.pattern = os.path.join(self.best_model_save_path, "*_best_test_rw_*.zip")

    def _on_step(self) -> bool:
        """Checks whether it's time to evaluate the model and handles the evaluation process.

        Returns:
            bool: True to continue training, False to stop (never stops training in this case).
        """
        self.steps_since_eval += 1

        # Perform evaluation every eval_freq steps
        if (self.num_timesteps > 400): # default: 10K
            if (self.eval_freq > 0 and self.n_calls % self.eval_freq == 0) or self.steps_since_eval >= self.eval_freq:
                infos = self.locals.get("infos", [])
                terminated = False
                truncated = False

                # Check all environments for termination or truncation (in case of multi-env)
                for idx, info in enumerate(infos):
                    if info.get("terminated", False):
                        terminated = True
                        logging.debug(f"Episode terminated in env {idx}")
                    if info.get("truncated", False):
                        truncated = True
                        logging.debug(f"Episode truncated in env {idx}")

                logging.info(f"terminated: {terminated}, truncated: {truncated}")

                # If episode is still running, then continue the training process until the current episode is finished
                if not (terminated or truncated):
                    logging.info(f"Waiting for episode to finish before doing the evaluation. Steps since last eval: {self.steps_since_eval}")
                    return True
                else:
                    logging.info("Episode is finished, starting the evaluation.")

                # Set evaluation environment: we use the tr5aining env, as we randomize all the elements of the scene for each episode, so there is
                # no need for using different envs
                self.eval_env = self.rl_manager.env

                # Evaluate policy over n_eval_episodes
                episode_rewards, _ = evaluate_policy(
                    self.model,
                    self.eval_env,
                    n_eval_episodes=self.n_eval_episodes,
                    render=self.render,
                    deterministic=True,
                    return_episode_rewards=True,
                    warn=False,
                )

                mean_reward = sum(episode_rewards) / len(episode_rewards)
                if self.verbose > 0:
                    logging.info(f"Evaluation at step {self.num_timesteps}: mean_reward={mean_reward:.2f}")

                # Save new best model if it outperforms previous best
                if self.best_model_save_path is not None and mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    model_name = os.path.basename(self.best_model_save_path)

                    # Remove any previous best models matching the pattern
                    for file_path in glob.glob(self.pattern):
                        try:
                            os.remove(file_path)
                            logging.info(f"Removed previous best (eval) model: {file_path}")
                        except OSError as e:
                            logging.error(f"Error: It was not possible to remove the file {file_path}: {e}")

                    # Save new best model with updated timestep
                    new_model_path = os.path.join(
                        self.best_model_save_path, f"{model_name}_best_test_rw_{self.num_timesteps}"
                    )
                    self.model.save(new_model_path)
                    logging.info(f"New best model saved: {new_model_path}")

                # Log evaluation result to CSV file if logging path is defined
                if self.log_path is not None:
                    log_file_path = os.path.join(self.log_path, "evaluations.csv")
                    with open(log_file_path, "a") as f:
                        f.write(f"{self.num_timesteps},{mean_reward}\n")
                    logging.info(f"Logged evaluation result to {log_file_path}")

                # Reset step counter since last evaluation
                self.steps_since_eval = 0

                # Log eval data
                self.logger.dump(self.num_timesteps)

        return True


def get_base_env(vec_env):
    env = vec_env
    if hasattr(env, 'envs'):  # DummyVecEnv or SubprocVecEnv
        env = env.envs[0]     # Access to the first environment
    while hasattr(env, 'venv'):  # Unwrap from SB3
        env = env.venv
    while hasattr(env, 'env'):  # Unwrap from gym
        env = env.env
    return env


class CustomMetricsCallback(BaseCallback):
    def __init__(self, rl_copp, total_timesteps, episodes_offset=0, sim_time_offset=0, eval_freq=4, verbose=0):
        super().__init__(verbose)
        self.rl_copp = rl_copp 
        self.eval_freq = eval_freq
        self.last_logged_episode = 0
        self.episode_count = 0
        self.episodes_offset = episodes_offset
        self.sim_time_offset = sim_time_offset
        self.total_timesteps= total_timesteps

        # Get base env
        self.base_env = get_base_env(self.rl_copp.env)
        logging.info(f"N_episodes: {self.base_env.n_ep}, ATO: {self.base_env.ato}")


    def _on_step(self) -> bool:
        # Get current episode number
        new_episode_count = self.base_env.n_ep

        # If the episode count has increased, increment it
        if self.episode_count != new_episode_count:
            self.episode_count = new_episode_count

        current_episode = self.base_env.n_ep+float(self.episodes_offset)
        current_sim_time = self.base_env.ato+float(self.sim_time_offset)

        logging.info(f"Current episode: {current_episode}, Current sim time: {current_sim_time}")

        if self.logger is None:
            raise RuntimeError("Logger not initialized CustomMetricsCallback")

        if self.episode_count != self.last_logged_episode and self.episode_count % self.eval_freq == 0:
            progress = int((self.num_timesteps / self.total_timesteps) * 100)
            print(f"Training Progress: {progress}%")

            # Logging into tensorboard
            logging.info(f"Logging custom metrics at timestep {self.num_timesteps}")
            self.logger.record("custom/sim_time", current_sim_time, self.num_timesteps)
            logging.info(f"Logging sim time: {current_sim_time} at timestep {self.num_timesteps}")
            # self.logger.record("custom/agent_time", base_env.total_time_elapsed, self.num_timesteps)
            self.logger.record("custom/episodes", current_episode, self.num_timesteps)
            logging.info(f"Logging episodes: {current_episode} at timestep {self.num_timesteps}")

            # Log the current episode count and simulation time
            self.logger.dump(self.num_timesteps)

            # Update the last logged episode to avoid redundant logging
            self.last_logged_episode = self.episode_count

        return True


# ------------------------------------
# ------------------------------------
# ------ Functions for testing -------
# ------------------------------------
# ------------------------------------


def init_metrics_test(env):
    """
    Function for getting the initial distance to the target, and the initial time of the episode.

    This function is called at the beginning of each episode during the testing process, so we can get the final metrics
    obtained during the test after finishing each episode.

    Args:
        env (gym): Custom environment to get the metrics from.

    Returns:
        None
    """
    env.initial_target_distance=env.observation[0]


def get_metrics_test(env):
    """
    Function for getting the all the desired metrics at the end of each episode, during the testing process.

    Args:
        env (gym): Custom environment to get the metrics from.

    Returns:
        initial_target_distance (float): Initial distance between the robot and the target.
        reached_target_distance (float): Final distance between the robot and the target obtained at the end of the episode.
        time_elapsed (float): Time counter to track the duration of the episode.
        reward (float): Reward obtained at the end of the episode.
        count (int): Total timesteps completed in the episode.
        
    """
    env.reached_target_distance=env.unwrapped.observation[0]
    return env.initial_target_distance,env.reached_target_distance,env.time_elapsed,env.reward, env.count, env.collision_flag, env.max_achieved, env.target_zone


def calculate_episode_distance(trajs_folder, traj_file_name):
    """
    Calculate the total distance traveled in a specific episode from its trajectory file.

    Args:
        trajs_folder (str): Path to the folder containing trajectory CSV files.
        traj_file_name (str): Name of the trajectory file (e.g., "trajectory_1.csv").

    Returns:
        float: Total distance traveled during the specified episode, calculated as the sum 
               of distances between successive positions in the trajectory.
    """
    current_traj_file = os.path.join(trajs_folder, traj_file_name)
    df_traj = pd.read_csv(current_traj_file)
    x_positions = df_traj["x"].values
    y_positions = df_traj["y"].values
    step_distances = np.sqrt(np.diff(x_positions)**2 + np.diff(y_positions)**2)
    distance_traveled = np.sum(step_distances)

    return distance_traveled


def calculate_average_distances(trajs_folder):
        """
        Calculate the average distance traveled per episode from the trajectory files.

        Args:
            trajs_folder (str): Path to the folder containing trajectory CSV files.

        Returns:
            float: Average distance traveled across all episodes.
        """
        traj_files = glob.glob(os.path.join(trajs_folder, "*.csv"))
        total_distance = 0.0
        count = 0
        if not traj_files:
            logging.warning(f"No trajectory files found in {trajs_folder}. Returning 0.0 as average distance.")
            return 0.0
        logging.info(f"Calculating average distance from {len(traj_files)} trajectory files in {trajs_folder}.")
        # Iterate through each trajectory file and calculate the distance
        # between successive positions
        for traj_file in traj_files:
            df_traj = pd.read_csv(traj_file)
            x_positions = df_traj["x"].values
            y_positions = df_traj["y"].values

            # Calculate the distance between successive positions
            step_distances = np.sqrt(np.diff(x_positions)**2 + np.diff(y_positions)**2)
            logging.info(f"Step distances for {traj_file}: {np.sum(step_distances):.2f} m")
            total_distance += np.sum(step_distances)
            count += 1

        return total_distance / count if count > 0 else 0.0


# ------------------------------------
# ------------------------------------
# ------ Functions for plotting ------
# ------------------------------------
# ------------------------------------


def get_data_for_spider(csv_path, args, column_names):
    """
    Extracts mean values of specific columns from rows in a CSV that match given model IDs.

    Args:
        csv_path (str): Path to the CSV file containing experiment data.
        args (argparse.Namespace): Parsed command-line arguments.
            - args.robot_name (str): Name of the robot to filter experiment names.
            - args.ids (list of int): List of model IDs to match at the end of experiment names.
        column_names (list): List of column headers to average for each matched model ID.

    Returns:
        dict: A dictionary where keys are model IDs (int) and values are pandas Series 
        containing the mean values of the specified columns. If no rows are found for a given ID, 
        the value is None.

    Example:
        If the CSV includes entries like 'robot1_model_34', 'robot2_model_34', and 'robot1_model_134',
        and you call the function with robot_name='robot1' and ids=[34, 134], the function returns
        a dictionary with averaged values for each matching ID.
    """
    # Cargar el CSV en un DataFrame de pandas
    try:
        df = pd.read_csv(csv_path)
        logging.info(f"CSV file loaded successfully from {csv_path}.")

    except Exception as e:
        logging.error(f"No csv file was found in {csv_path}. Exception: {e}")
        sys.exit()
    
    data_to_extract = {}

    df_filtered = df[df.iloc[:, 0].notna()]
    # Process each ID in the args.model_ids list
    for id in args.model_ids:
        # Search rows in the first column which finish with the provided ID      

        pattern = re.compile(rf'^{re.escape(args.robot_name)}_model_{id}(?:_|$)')
        filter = df_filtered.iloc[:, 0].apply(lambda x: bool(pattern.match(str(x))))
        filtered_rows = df_filtered[filter]
        
        # If no row is found, then assign None
        if filtered_rows.empty:
            data_to_extract[id] = None
        else:
            # Select the desired columns and calculte the mean
            data = filtered_rows[column_names].mean(axis=0)
            data_to_extract[id] = data

        logging.info(f"Data extracted for ID {id}: {data_to_extract[id]}")
    
    return data_to_extract


def process_spider_data (df, tolerance=0.05):
    """
    Extracts data from the train and test dataframes and normalizes those metrics for radar chart visualization, ensuring:
    - Min-Max Scaling for standard metrics.
    - Inverse scaling for loss metrics (Min-Max on absolute value).
    - Inverse scaling for `rollout/ep_len_mean` to prioritize lower values.
    - A tolerance is applied to avoid exact 0 or 1 in the normalized values.

    Args:
        df (DataFrame): Dataframe with data for each ID as pandas.Series.
        tolerance (float, Optional): Percentage tolerance applied to prevent normalization from reaching 0 or 1.

    Returns:
        data_list (list of lists): Normalized metric values for each ID.
        names (list): List of experiment names formatted as "T_<action_time>".
        labels (list): List of metric names.
    """
    data_list = []
    names = []

    # Extract metric labels excluding "Action time (s)"
    labels = df.drop(columns=["Action time (s)"]).columns.tolist()

    # Separate the different metrics depending on how they work:
    negative_metrics = [col for col in labels if "actor_loss" in col.lower()]  # More negative values --> Better
    min_metrics = [col for col in labels if any(metric in col.lower() for metric in ["time", "critic_loss", "ep_len_mean", "distance"])]    # Smaller values (closer to 0) --> Better
    max_metrics = [col for col in labels if col not in negative_metrics + min_metrics]    # Bigger values --> Better

    # Normalize data
    df_normalized = df.copy()

    # --- max_metrics = ['rollout/ep_rew_mean', 'Avg reward', 'Target zone 3 (%)']
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
        
    # --- min_metrics = ['train/critic_loss', 'rollout/ep_len_mean', 'Avg time reach target', 'Avg episode distance (m)]
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
        names.append(f"{action_time:.2f}s")
        data_list.append(df_normalized.loc[id_, labels].tolist())

    return data_list, names, labels


def plot_multiple_spider(data_list, labels, names, title='Models Comparison'):
    """
    Plots multiple spider charts on the same figure to compare different models.

    Args:
        data_list (list of lists): A list of several lsit of metrics, one per model.
        labels (list): List of labels for the axes (metrics).
        names (list): List of names corresponding to each dataset (for the legend). They correspond to the action time in seconds.
        title (str, Optional): The title of the chart.
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


def moving_average(data, window_size=10):
    """
    Applies a moving average filter to smooth the data.

    Args:
        data (array-like): Sequence of numeric values to be smoothed.
        window_size (int): Number of points to include in each averaging window.

    Returns:
        np.ndarray: The smoothed data as a NumPy array.
    """
    return pd.Series(data).rolling(window=window_size, center=True).mean().to_numpy()


def get_color_map(n_colors):
    """
    Returns a list of colors using tab20 first, then filling with tab20b and tab20c if needed.

    Args:
        n_colors (int): Total number of colors needed.

    Returns:
        list: List of RGBA tuples.
    """
    tab20 = plt.cm.get_cmap('tab20')
    tab20b = plt.cm.get_cmap('tab20b')
    tab20c = plt.cm.get_cmap('tab20c')

    colors = [tab20(i) for i in range(20)]  # First 20 from tab20

    extra_needed = max(0, n_colors - 15)
    half = (extra_needed + 1) // 2  # Divide extras equally (first goes to tab20b)

    colors += [tab20b(i) for i in range(half)]
    colors += [tab20c(i) for i in range(extra_needed - half)]

    return colors[:n_colors]


def get_legend_columns(n_models, items_per_column=4):
    """
    Computes the number of columns for the legend based on the number of models.

    Args:
        n_models (int): Total number of models (legend items).
        items_per_column (int): Maximum number of items per column.

    Returns:
        int: Recommended number of legend columns.
    """
    return max(1, (n_models + items_per_column - 1) // items_per_column)


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


def delayed_exponential_model(x, A, k, B, delay):
    """Modelo exponencial con retardo
    y = B para x < delay
    y = A * (1 - exp(-k * (x - delay))) + B para x >= delay
    """
    result = np.zeros_like(x)
    mask = x < delay
    result[mask] = B
    result[~mask] = A * (1 - np.exp(-k * (x[~mask] - delay))) + B
    return result


def delayed_exponential_derivative(x, A, k, B, delay):
    """Derivada del modelo exponencial con retardo"""
    result = np.zeros_like(x)
    mask = x < delay
    result[mask] = 0
    result[~mask] = A * k * np.exp(-k * (x[~mask] - delay))
    return result


def plot_metric_boxplot_by_timestep(df, metric, ylabel, color='#2678b0'):
    """
    Plot a boxplot for a continuous metric with the X-axis showing the actual timestep (float),
    ordered numerically.

    Args:
        df (pd.DataFrame): DataFrame containing all models' testing results.
        metric (str): Name of the metric to be plotted (must be a column in `df`).
        ylabel (str): Label for the Y-axis.
        color (str): Color used for the boxplot fill.
    """
    # Extract unique models (e.g., '0.4s') and corresponding timestep values
    unique_models = sorted(df["Model"].unique(), key=lambda x: float(x.replace('s', '')))
    timesteps_ordered = [float(model.replace('s', '')) for model in unique_models]

    box_data = []
    valid_timesteps = []

    for i, model in enumerate(unique_models):
        model_data = df[df["Model"] == model][metric].values
        clean_data = model_data[np.isfinite(model_data)]
        if len(clean_data) > 0:
            box_data.append(clean_data)
            valid_timesteps.append(timesteps_ordered[i])

    if not box_data:
        logging.warning(f"[!] No valid data found for metric '{metric}'")
        return

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    box_width = min(np.diff(valid_timesteps)) * 0.7 if len(valid_timesteps) > 1 else 0.1
    bp = ax.boxplot(box_data, positions=valid_timesteps, widths=box_width, patch_artist=True)

    # Customize box appearance
    for patch in bp['boxes']:
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
        patch.set_alpha(1)

    for whisker in bp['whiskers']:
        whisker.set_color('black')
        whisker.set_linewidth(1)

    for cap in bp['caps']:
        cap.set_color(color)
        cap.set_linewidth(1)

    for median in bp['medians']:
        median.set_color('#fe7c2b')
        median.set_linewidth(1)

    for flier in bp['fliers']:
        flier.set(marker='o', color='black', alpha=0.3)

    # X-axis configuration
    ax.set_xticks(valid_timesteps)
    ax.set_xlim(min(valid_timesteps) - 0.2, max(valid_timesteps) + 0.1)
    ax.set_xticklabels([f"{t}" for t in valid_timesteps], fontsize=14, rotation=0)

    # Labels and grid
    ax.set_xlabel("Timestep (s)", fontsize=20, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=20, labelpad=10)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid(True)
    plt.tight_layout()

    return fig, ax


def get_convergence_point(file_path, x_axis, convergence_threshold=0.02):
    """
    Analyze convergence for trained models.

    Args:
        file_path (str): Path to the CSV file containing training data.
        x_axis (str): Name of the x-axis to analyze. Must be one of 
            'WallTime', 'Steps', 'SimTime', or 'Episodes'.
        convergence_threshold (float): Threshold for determining convergence. 
            Defaults to 0.02.

    Returns:
        tuple: A tuple containing:
            - convergence_point (float): The value on the x-axis where convergence occurs.
            - reward_fit (np.ndarray): The fitted reward curve.
            - x_raw (np.ndarray): Raw x-axis values from the CSV.
            - reward (np.ndarray): Raw reward values from the CSV.
            - reward_at_convergence (float): The reward value at the convergence point.
    """
    # Read csv file
    df = pd.read_csv(file_path)

    # Prepare x axis depending on the selected option
    if x_axis == "WallTime":
        df['Step timestamp'] = pd.to_datetime(df['Step timestamp'], format='%Y-%m-%d_%H-%M-%S')
        start_time = df['Step timestamp'].iloc[0]
        df['Relative time'] = (df['Step timestamp'] - start_time).dt.total_seconds() / 3600
        x_raw = df['Relative time'].values
    elif x_axis == "Steps":
        x_raw = df['Step'].values
        x_raw = x_raw - x_raw[0]  # para que empiece en 0
    elif x_axis == "SimTime":
        x_raw = df['custom/sim_time'].values
        x_raw = (x_raw - x_raw[0]) / 3600  # convertir a horas
    elif x_axis == "Episodes":
        x_raw = df['custom/episodes'].values
        x_raw = x_raw - x_raw[0]
    else:
        raise ValueError("x_axis debe ser 'WallTime', 'Steps', 'SimTime', o 'Episodes'")
    
    # Get reward (y axis)
    reward = df['rollout/ep_rew_mean'].values
        
    
    # Normalize x axis
    x_norm = (x_raw - np.min(x_raw)) / (np.max(x_raw) - np.min(x_raw))
    
    # As there can be some confusing data at the beggining, jsut skip the first start_fraction of the data
    start_fraction=0.001  # For method 1, change this to 0.05 and uncomment method1 code (and comment method 2)
    start_idx = int(len(x_norm) * start_fraction)
    x_norm_window = x_norm[start_idx:]
    
    reward_window = reward[start_idx:]
    
    # Estimate initial delay
    min_idx = np.argmin(reward_window)
    delay_estimate = x_norm_window[min_idx]


    initial_estimation = [
        np.max(reward_window) - np.min(reward_window),  # A
        1.0,                                             # k
        np.min(reward_window),                          # B
        delay_estimate                                   # delay
    ]
    
    # Adjust exponential model with delay
    popt, _ = curve_fit(delayed_exponential_model, x_norm_window, reward_window, p0=initial_estimation)
    A, k, B, delay = popt

    # Generate model
    reward_fit = delayed_exponential_model(x_norm, A, k, B, delay)
    reward_derivative = delayed_exponential_derivative(x_norm, A, k, B, delay)

    logging.debug("Parameters of the exponential model with delay")
    logging.debug(f"A: {A}")
    logging.debug(f"k: {k}")
    logging.debug(f"B: {B}")
    logging.debug(f"delay: {delay}")
    
    # Find the point in the x-axis when the derivative crosses below the threshold or zero
    # Method 1: skipping first points

    for i in range(start_idx, len(reward_derivative)):
        if np.abs(reward_derivative[i]) < convergence_threshold:
            convergence_point_norm = x_norm[i]
            break
    else:
        convergence_point_norm = x_norm[-1]

    # Method 2: dynamic window to avoid minimum or maximum locals
    window_size = round(0.2*len(reward_derivative))
    convergence_point_norm = x_norm[-1] # default value
    for i in range(len(reward_derivative) - window_size):
        window = reward_derivative[i:i + window_size]
        if np.all(np.abs(window) < convergence_threshold):
            convergence_point_norm = x_norm[i]
            break
    
    # Convert to original scale
    convergence_point = convergence_point_norm * (np.max(x_raw) - np.min(x_raw)) + np.min(x_raw)
    
    # Get nearest index to convergence point
    idx_convergence = np.argmin(np.abs(x_raw - convergence_point))
    reward_at_convergence = reward[idx_convergence]
    
    return convergence_point, reward_fit, x_raw, reward, reward_at_convergence


def plot_metrics_comparison_smooth_with_original_deprecated(rl_copp_obj, metric, title="Comparison"):
    """
    Plot both raw and smoothed metric curves of multiple models for visual comparison.

    Args:
        rl_copp_obj (RLCoppeliaManager): Instance of RLCoppeliaManager class for managing paths and arguments.
        metric (str): The metric to be plotted ("rewards" or "episodes_length").
        title (str): Title of the plot.
    """
    smooth_flag = True
    smooth_level = 50  # Size of moving average window

    training_metrics_path = rl_copp_obj.paths["training_metrics"]
    train_records_csv_name = os.path.join(training_metrics_path, "train_records.csv")
    timestep_to_data = {}

    for model_index in range(len(rl_copp_obj.args.model_ids)):
        model_id = rl_copp_obj.args.model_ids[model_index]
        file_pattern = f"{rl_copp_obj.args.robot_name}_model_{model_id}_*.csv"
        files = glob.glob(os.path.join(rl_copp_obj.base_path, "robots", rl_copp_obj.args.robot_name, "training_metrics", file_pattern))

        if not files:
            logging.warning(f"No CSV found for model {model_id}. Skipping.")
            continue

        try:
            df = pd.read_csv(files[0])
        except Exception as e:
            logging.error(f"Could not read file for model {model_id}. Error: {e}")
            continue

        model_name = f"{rl_copp_obj.args.robot_name}_model_{model_id}"
        timestep = get_data_from_training_csv(model_name, train_records_csv_name, column_header="Action time (s)")

        steps = df['Step'].values
        if metric == "rewards":
            data = df['rollout/ep_rew_mean'].values
        elif metric == "episodes_length":
            data = df['rollout/ep_len_mean'].values
        else:
            logging.error(f"Unknown metric: {metric}")
            return

        smoothed_data = moving_average(data, window_size=smooth_level) if smooth_flag else data
        smoothed_steps = steps[:len(smoothed_data)]

        if timestep not in timestep_to_data:
            timestep_to_data[timestep] = []

        timestep_to_data[timestep].append((steps, data, smoothed_steps, smoothed_data))

    color_map = plt.cm.get_cmap("tab10", len(timestep_to_data))
    plt.figure(figsize=(13, 10))

    for idx, (timestep, series_list) in enumerate(timestep_to_data.items()):
        for steps, raw_data, smooth_steps, smooth_data in series_list:
            color = color_map(idx)
            plt.plot(steps, raw_data, label=f"Raw Model {timestep}s", linestyle=':', alpha=0.5, color=color)
            plt.plot(smooth_steps, smooth_data, label=f"Smoothed Model {timestep}s", linestyle='-', linewidth=2, color=color)

    plt.xlabel('Steps', fontsize=20)
    plt.ylabel(metric.capitalize(), fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=14, ncol=2)
    plt.grid(True)
    plt.tight_layout()

    if rl_copp_obj.args.save_plots:
        filename = f"metrics_comparison_{metric}.png"
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()




def find_otherdata_files(rl_copp_obj, model_index: int) -> List[str]:
    """Find 'otherdata' CSV files for a given model index.

    Args:
        rl_copp_obj: RLCoppeliaManager-like object containing paths and args.
        model_index (int): Index in rl_copp_obj.args.model_ids.

    Returns:
        list[str]: List of CSV paths.
        model_name (str): Name of the model.
    """
    model_id = rl_copp_obj.args.model_ids[model_index]
    model_name = f"{rl_copp_obj.args.robot_name}_model_{model_id}"
    robot_name = rl_copp_obj.args.robot_name

    
    file_pattern = f"{model_name}_*_otherdata_*.csv"
    subfolder_pattern = f"{model_name}_*_testing"
    files = glob.glob(os.path.join(
        rl_copp_obj.base_path, "robots", robot_name, "testing_metrics",
        subfolder_pattern, file_pattern
    ))
    logging.info(f"Files found: {files}")
    if rl_copp_obj.args.csv_file_name is not None:
        f = rl_copp_obj.args.csv_file_name
        
        full_path = next((path for path in files if os.path.basename(path) == f), None)
        logging.info(f"File {full_path} specified as argument is part of the files.")

        # file could be str, list[str] o list[list[str]]; so let's flatten it.
        if isinstance(full_path, str):
            files = [full_path]
        else:
            files = []
            for item in full_path:
                if isinstance(item, (list, tuple, set)):
                    files.extend(list(item))
                else:
                    files.append(item)
                    
    return files, model_name


def _flatten_files_arg(files: Union[str, Iterable]) -> List[str]:
    """Return a flat list of paths from str | list[str] | list[list[str]]."""
    if isinstance(files, str):
        return [files]
    flat: List[str] = []
    for item in files:
        if isinstance(item, (list, tuple, set)):
            flat.extend(list(item))
        else:
            flat.append(item)
    return flat



def load_and_concat_otherdata(files: Union[str, Iterable]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load multiple 'otherdata' CSV files, stack them vertically and renumber episodes globally.

    It preserves the original episode number per file in `episode_original`, the file name in
    `source_file`, and assigns a running `episode_global` across files: the second file starts
    at (last_episode_of_first + 1), and so on.

    Args:
        files: A path string, list of paths, or nested lists of paths.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - df_all: concatenated dataframe with added columns:
                ['source_file', 'episode_original', 'episode_global']
            - episode_map: unique mapping rows with columns:
                ['source_file', 'episode_original', 'episode_global']
    """
    paths = _flatten_files_arg(files)
    if not paths:
        raise FileNotFoundError("No otherdata CSV files provided.")

    all_dfs: List[pd.DataFrame] = []
    episode_offset = 0
    map_rows = []

    for p in paths:
        logging.info(f"Reading file: {p}")
        df = pd.read_csv(p)

        if "Episode number" not in df.columns:
            raise KeyError(f"'Episode number' column not found in: {p}")

        # Keep original episode column
        df["source_file"] = os.path.basename(p)
        df["episode_original"] = df["Episode number"].astype(int)

        # Compute global episode number as original + offset
        df["episode_global"] = df["episode_original"] + episode_offset

        # For the mapping table, one row per (file, original) is enough
        map_rows.append(
            df[["source_file", "episode_original", "episode_global"]]
            .drop_duplicates()
            .copy()
        )

        # Update offset for the next file (use max original episode of this file)
        max_ep_this = int(df["episode_original"].max()) if len(df) else 0
        episode_offset += max_ep_this

        all_dfs.append(df)

    df_all = pd.concat(all_dfs, ignore_index=True)
    episode_map = pd.concat(map_rows, ignore_index=True).sort_values(
        ["episode_global", "source_file", "episode_original"]
    ).reset_index(drop=True)

    return df_all, episode_map


def drop_single_step_episodes(df_all: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Drop episodes that contain only one row (single timestep) and log what was removed.

    The dataframe must contain 'episode_global', 'episode_original', and 'source_file' columns.

    Args:
        df_all: Concatenated dataframe produced by `load_and_concat_otherdata`.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - df_filtered: dataframe without single-step episodes.
            - removed_info: dataframe listing removed episodes with columns:
                ['source_file', 'episode_original', 'episode_global', 'rows_in_episode']
    """
    required = {"episode_global", "episode_original", "source_file"}
    missing = required.difference(df_all.columns)
    if missing:
        raise KeyError(f"Missing required columns in df_all: {sorted(missing)}")

    # Count rows per episode (global)
    counts = df_all.groupby("episode_global").size().rename("rows_in_episode")
    single_ep_ids = counts[counts == 1].index.tolist()

    if not single_ep_ids:
        logging.info("No single-step episodes found. Nothing to drop.")
        return df_all, pd.DataFrame(columns=["source_file", "episode_original", "episode_global", "rows_in_episode"])

    # Build removal info with file + original id for logging/reporting
    removed_rows = (
        df_all[df_all["episode_global"].isin(single_ep_ids)]
        .loc[:, ["source_file", "episode_original", "episode_global"]]
        .copy()
    )
    removed_rows = removed_rows.merge(
        counts.reset_index(), on="episode_global", how="left"
    )

    # Log per file the removed episodes
    for src, sub in removed_rows.groupby("source_file"):
        eps = sub.sort_values("episode_original")["episode_original"].tolist()
        logging.info(f"Removed {len(eps)} single-step episodes from '{src}': {eps}")

    # Drop single-step episodes
    df_filtered = df_all[~df_all["episode_global"].isin(single_ep_ids)].reset_index(drop=True)

    return df_filtered, removed_rows


def preprocess_otherdata_files(rl_copp_obj, model_index):
    files, model_name = find_otherdata_files(rl_copp_obj, model_index)
    df_all, _episode_map = load_and_concat_otherdata(files)
    df_filtered, _removed_info = drop_single_step_episodes(df_all)

    return df_filtered, model_name


def detect_columns(df: pd.DataFrame, params_env: Optional[Dict] = None) -> Dict[str, object]:
    """Detect important columns (timestep, obs names, laser list, distance).

    Tries to use `params_env` when provided; otherwise falls back to pattern-based
    detection (distance, angle, and any columns prefixed with 'laser_obs').

    Args:
        df: DataFrame to inspect.
        params_env: Optional dict with hints:
            - 'action_names' (list[str])
            - 'observation_names' (list[str])

    Returns:
        dict with keys:
          - "timestep_col": str
          - "observation_cols": list[str] (ordered: lasers..., distance, angle)
          - "laser_cols": list[str]
          - "distance_col": Optional[str]
    """
    cols = list(df.columns)

    # --- Timestep column detection ---
    action_names = (params_env or {}).get("action_names", [])
    timestep_candidates = ["timestep", "action_time", "dt", "time_step"]

    timestep_col = None
    if action_names:
        # Prefer an action name that clearly denotes timestep and exists in df
        for nm in action_names:
            if nm in df.columns and nm.lower() in ("timestep", "action_time", "dt", "time_step"):
                timestep_col = nm
                break
    if timestep_col is None:
        timestep_col = next((c for c in timestep_candidates if c in cols), None)

    if timestep_col is None:
        raise ValueError("Could not detect timestep column. Ensure your CSV has a 'timestep' column "
                         "or pass action_names in params_env.")

    # --- Observation detection ---
    # Prefer params_env["observation_names"] if available
    obs_names_param = (params_env or {}).get("observation_names", [])
    if obs_names_param:
        observation_cols = [c for c in obs_names_param if c in cols]
    else:
        # Heuristic: any 'laser_obs*' + 'distance' + 'angle' if present
        observation_cols = []
        for c in cols:
            if c.startswith("laser_obs"):
                observation_cols.append(c)
        if "distance" in cols:
            observation_cols.append("distance")
        if "angle" in cols:
            observation_cols.append("angle")

    # Order: lasers sorted by index, then distance, then angle
    laser_cols = sorted([c for c in observation_cols if c.startswith("laser_obs")], key=laser_index_key)
    distance_col = "distance" if "distance" in observation_cols else None
    angle_col = "angle" if "angle" in observation_cols else None
    ordered_obs = laser_cols + ([distance_col] if distance_col else []) + ([angle_col] if angle_col else [])

    return {
        "timestep_col": timestep_col,
        "observation_cols": ordered_obs,
        "laser_cols": laser_cols,
        "distance_col": distance_col,
    }


def laser_index_key(name: str) -> int:
    """Numerical sorting key for 'laser_obs<N>'."""
    try:
        return int(name.replace("laser_obs", ""))
    except Exception:
        return 10**9  # unknowns go last


def edges_to_bin_labels(edges: List[float]) -> List[str]:
    """Build human-friendly bin labels like '0.2-0.5 s' for a list of edges.

    Args:
        edges: Monotonically increasing edges of length >= 2.

    Returns:
        List of length (len(edges)-1) with pretty labels.
    """
    labels = []
    for i in range(len(edges) - 1):
        a = float(edges[i])
        b = float(edges[i + 1])
        labels.append(f"{a:g}–{b:g} s")
    return labels


def clean_and_bin_timesteps(
    df: pd.DataFrame,
    timestep_col: str,
    bins: Tuple[float, ...] | List[float],
) -> pd.DataFrame:
    """Sanitize timestep column and build categorical bins for ANY number of ranges.

    - Converts the timestep column to float.
    - Drops rows with NaN timesteps.
    - Filters to [bins[0], bins[-1]].
    - Adds:
        * 'timestep_bin' (categorical label string)
        * 'timestep_bin_id' (0..N-1 for N bins)

    Args:
        df: DataFrame with a timestep column.
        timestep_col: Name of the timestep column (e.g., 'timestep').
        bins: Sequence of edges, length >= 2. N bins are formed by N+1 edges.

    Returns:
        Copy of df with bin columns added.
    """
    if not isinstance(bins, (list, tuple)) or len(bins) < 2:
        raise ValueError("`bins` must be a list/tuple of at least 2 edges (monotonically increasing).")

    # Ensure monotonic increasing and unique edges
    edges = [float(x) for x in bins]
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            raise ValueError("`bins` edges must be strictly increasing.")

    out = df.copy()
    out[timestep_col] = pd.to_numeric(out[timestep_col], errors="coerce")
    out = out.dropna(subset=[timestep_col])

    lo, hi = edges[0], edges[-1]
    mask = (out[timestep_col] >= lo) & (out[timestep_col] <= hi)
    dropped = (~mask).sum()
    if dropped:
        logging.info(f"Dropping {dropped} rows with timesteps out of range [{lo}, {hi}].")
    out = out.loc[mask].copy()

    labels = edges_to_bin_labels(edges)
    out["timestep_bin"] = pd.cut(
        out[timestep_col],
        bins=edges,
        labels=labels,
        include_lowest=True,
        right=True,
    )
    out["timestep_bin_id"] = out["timestep_bin"].cat.codes  # 0..N-1

    return out


def summarize_timestep_bins(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Count absolute and relative frequencies of timestep bins."""
    counts = df["timestep_bin"].value_counts().sort_index()
    percent = counts / counts.sum()
    logging.info("Timestep bin usage:\n" + counts.to_string())
    return counts, percent


def summarize_state_means_by_bin(
    df: pd.DataFrame,
    observation_cols: List[str],
    distance_col: Optional[str],
    laser_cols: List[str],
) -> pd.DataFrame:
    """Compute per-bin means of selected state features (distance/min_laser/angle)."""
    feats = []
    if distance_col:
        feats.append("_distance")
    if "min_laser" in df.columns and len(laser_cols) > 0:
        feats.append("min_laser")
    if "angle" in observation_cols and "angle" in df.columns:
        feats.append("angle")

    if not feats:
        return pd.DataFrame()

    means = df.groupby("timestep_bin")[feats].mean()
    logging.info("Per-bin state means:\n" + means.to_string())
    return means


def spearman_correlations(
    df: pd.DataFrame,
    timestep_col: str,
    observation_cols: List[str],
) -> pd.DataFrame:
    """Compute Spearman correlation between timestep and each observation column."""
    res = []
    for col in observation_cols:
        if col not in df.columns:
            continue
        x = pd.to_numeric(df[timestep_col], errors="coerce")
        y = pd.to_numeric(df[col], errors="coerce")
        m = x.notna() & y.notna()
        if m.sum() < 3:
            continue
        r, p = spearmanr(x[m], y[m])
        res.append({"variable": col, "spearman_r": float(r), "p_value": float(p)})
    out = pd.DataFrame(res).set_index("variable").sort_values("spearman_r", ascending=False)
    logging.info("Spearman correlations (timestep vs obs):\n" + out.to_string())
    return out


def timestep_on_episode_last_step(df: pd.DataFrame, timestep_col: str) -> pd.Series:
    """Distribution of timestep bins on the last row of each episode."""
    if "Episode number" not in df.columns:
        logging.info("No 'Episode number' column; skipping last-step analysis.")
        return pd.Series(dtype=int)
    last_idx = df.groupby("Episode number").tail(1).index
    last_bins = df.loc[last_idx, "timestep_bin"]
    counts = last_bins.value_counts().sort_index()
    logging.info("Last-step timestep bins:\n" + counts.to_string())
    return counts


def near_collision_analysis(
    df: pd.DataFrame,
    min_laser_col: str,
    timestep_bin_col: str,
    params_env: Optional[Dict],
    laser_tolerance: float,
) -> Dict[str, object]:
    """Near-collision analysis using min_laser < (max_crash_dist + tolerance)."""
    max_crash = (params_env or {}).get("max_crash_dist", 0.18)
    thr = max_crash + float(laser_tolerance)

    if min_laser_col not in df.columns:
        logging.info("No min_laser column; skipping near-collision analysis.")
        return {"threshold": thr, "global_risk_share": np.nan, "bin_risk_share": pd.Series(dtype=float), "counts": pd.DataFrame()}

    risk_mask = pd.to_numeric(df[min_laser_col], errors="coerce") < thr
    global_share = float(risk_mask.mean())

    grp = df.groupby(timestep_bin_col)
    per_bin_risk = grp.apply(lambda g: (pd.to_numeric(g[min_laser_col], errors="coerce") < thr).mean())

    counts_df = pd.DataFrame({
        "total": grp.size(),
        "risk_share": per_bin_risk
    }).sort_index()

    logging.info(f"Near-collision threshold: {thr:.3f} m / Global % of steps in which the robot is under risk: {global_share:.3%}")
    logging.info("Near-collision risk by timestep bin:\n" + per_bin_risk.to_string())

    return {
        "threshold": thr,
        "global_risk_share": global_share,
        "bin_risk_share": per_bin_risk.sort_index(),
        "counts": counts_df,
    }


def clean_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize typical column names from logs into canonical names."""
    df = df.copy()
    df.columns = [c.strip().replace("\ufeff","") for c in df.columns]
    m = {}
    for c in df.columns:
        lc = c.lower().replace(" ", "").replace("_","")
        if lc in ("positionidx","posidx","positionindex"):
            m[c] = "Position idx"
        elif lc in ("scenarioidx","scidx","scenarioindex"):
            m[c] = "Scenario idx"
        elif lc in ("trialidx","trial","trialindex"):
            m[c] = "Trial idx"
        elif lc in ("posx","x"):
            m[c] = "Pos X"
        elif lc in ("posy","y"):
            m[c] = "Pos Y"
        elif lc in ("timestep","time_step","dt","t"):
            m[c] = "Timestep"
    return df.rename(columns=m) if m else df

def stat_sanity(a: np.ndarray, how: str) -> float:
    funcs = {"median": np.median, "mean": np.mean, "min": np.min, "max": np.max, "std": np.std}
    if how not in funcs:
        raise ValueError("stat must be one of 'median','mean','min','max','std'")
    return float(funcs[how](a))

def idw(centers_xy: np.ndarray,
        pts_xy: np.ndarray,
        vals: np.ndarray,
        power: int = 2,
        k: int = 6,
        eps: float = 1e-9):
    """
    Interpolate scalar values from scattered 2D samples using Inverse Distance Weighting (IDW).

    Given N sample points with known scalar values and M query points (cell centers),
    this function estimates a value at each query point as a distance-weighted average
    of the k nearest samples. Closer samples contribute more than farther ones; the
    weighting steepness is controlled by `power`.

    The implementation uses a KD-tree (scipy.spatial.cKDTree) for efficient neighbor
    searches and returns both the interpolated values and the minimum neighbor distance
    per query (useful for masking areas far from any sample).

    Parameters
    ----------
    centers_xy : np.ndarray, shape (M, 2)
        Query points where you want interpolated values (e.g., grid cell centers).
        Each row is (x, y) in the same coordinate system/units as `pts_xy`.

    pts_xy : np.ndarray, shape (N, 2)
        Sample 2D coordinates. Each row is (x, y). Must not be empty.
        If N < k, the function automatically reduces `k` to N.

    vals : np.ndarray, shape (N,)
        Scalar value at each sample point in `pts_xy`. Must align one-to-one with rows of `pts_xy`.

    power : int, default=2
        IDW power/exponent p in the weight formula w = 1 / (d^p).
        Larger values make the interpolation more local (the nearest points dominate more).
        Typical choices are 1, 2, or 3.

    k : int, default=6
        Number of nearest neighbors to use for each query. Internally clamped to `min(k, N)`.
        Use k=1 to emulate nearest-neighbor assignment (no averaging).

    eps : float, default=1e-9
        Small positive value added inside the distance to avoid division by zero when a query
        lies exactly on a sample (i.e., replaces d with max(d, eps)). If a query coincides with
        a sample, the result will be ~that sample’s value (within numerical precision).

    Returns
    -------
    interp : np.ndarray, shape (M,)
        Interpolated scalar value at each query in `centers_xy`, computed as the normalized
        weighted average of the k nearest sample values.

    dmin : np.ndarray, shape (M,)
        The minimum distance from each query to its nearest neighbor among the k used.
        Commonly used to mask queries that are “too far” from any sample (e.g., dmin > radius).

    Notes
    -----
    - The weight for neighbor i at distance d_i is w_i = 1 / (max(d_i, eps) ** power).
      The returned value is sum(w_i * v_i) / sum(w_i).
    - Units matter: distances are computed in the same units as `pts_xy` and `centers_xy`.
      If your coordinates are in meters, the effective spatial influence of `power` and `k`
      depends on the spatial density of points (e.g., 0.2 m vs 1.0 m spacing).
    - If `k` equals 1, the method degenerates to nearest-neighbor assignment (no smoothing).
    - If `N == 0`, this routine is undefined; ensure you have at least one sample.
    - Performance: building the KD-tree is O(N log N); each query is ~O(log N + k).


    """
    tree = cKDTree(pts_xy)
    k = min(k, len(pts_xy))
    dists, idxs = tree.query(centers_xy, k=k)
    d = np.maximum(dists, eps)
    w = 1.0 / (d ** power)
    if w.ndim == 1:  # k==1
        num = w * vals[idxs]
        den = w
    else:
        num = (w * vals[idxs]).sum(axis=1)
        den = w.sum(axis=1)
    return (num / den), dists.min(axis=1)


def plot_and_maybe_save_hist(counts: pd.Series, title: str, save_path: str | None):
        """
        Replicates a simple bar chart like utils.plot_timestep_usage_hist but also saves if path is given.
        """
        if counts.empty:
            return
        fig = plt.figure(figsize=(7, 4.2))
        ax = fig.add_subplot(111)
        ax.bar(counts.index.astype(str), counts.values)
        ax.set_title(title)
        ax.set_ylabel("count")
        ax.set_xlabel("timestep bin")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)


def plot_timestep_usage_hist(counts: pd.Series, title: str = "") -> None:
    """Simple bar chart of timestep bin counts."""
    if counts.empty:
        return
    plt.figure(figsize=(7, 4.2))
    plt.bar(counts.index.astype(str), counts.values)
    plt.title(title)
    plt.ylabel("count")
    plt.xlabel("timestep bin")
    plt.tight_layout()


def plot_last_step_hist(counts: pd.Series, title: str = "") -> None:
    """Bar chart for last-step timestep bins."""
    if counts.empty:
        return
    plt.figure(figsize=(7, 4.2))
    plt.bar(counts.index.astype(str), counts.values)
    plt.title(title)
    plt.ylabel("count (episodes)")
    plt.xlabel("timestep bin (last step)")
    plt.tight_layout()


def plot_near_collision_hist(share: pd.Series, title: str = "") -> None:
    """Bar chart of near-collision share per timestep bin."""
    if share.empty:
        return
    plt.figure(figsize=(7, 4.2))
    plt.bar(share.index.astype(str), share.values)
    plt.title(title)
    plt.ylabel("near-collision share")
    plt.xlabel("timestep bin")
    plt.ylim(0, 1)
    plt.tight_layout()


def plot_hexbin_mean_timestep(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    c_col: str,
    gridsize: int = 60,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    clabel: str = "Mean timestep",
) -> None:
    """2D interpretative map: mean timestep over (x,y) via hexbin.

    This offers a legible "policy map" of how timestep changes across state-space.
    """
    X = pd.to_numeric(df[x_col], errors="coerce")
    Y = pd.to_numeric(df[y_col], errors="coerce")
    C = pd.to_numeric(df[c_col], errors="coerce")
    m = X.notna() & Y.notna() & C.notna()

    if m.sum() < 10:
        logging.info("Not enough valid points for hexbin map; skipping.")
        return

    plt.figure(figsize=(7.8, 6.2))
    hb = plt.hexbin(
        X[m].values, Y[m].values, C=C[m].values,
        reduce_C_function=np.mean, gridsize=gridsize, linewidths=0.0
    )
    cb = plt.colorbar(hb)
    cb.set_label(clabel)
    plt.xlabel(xlabel or x_col)
    plt.ylabel(ylabel or y_col)
    plt.title(title)
    plt.tight_layout()


def plot_violin_timestep_by_quantiles(
    df: pd.DataFrame,
    value_col: str,
    bin_on_col: str,
    q_edges: Tuple[float, float, float, float, float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
) -> None:
    """Violin plot of `value_col` (timestep) vs quantile bins of `bin_on_col`.

    Example uses:
      - value_col='timestep', bin_on_col='distance'
      - value_col='timestep', bin_on_col='min_laser'

    Args:
        df: DataFrame with both columns.
        value_col: Name of the numeric column to plot (timestep).
        bin_on_col: Column to bin by quantiles (distance or min_laser).
        q_edges: Quantile edges, e.g., (0, .25, .5, .75, 1).
        title/xlabel/ylabel: Plot labels.
    """
    if bin_on_col not in df.columns or value_col not in df.columns:
        logging.info(f"Skipping violin plot: missing columns ({bin_on_col}, {value_col}).")
        return

    X = pd.to_numeric(df[bin_on_col], errors="coerce")
    Y = pd.to_numeric(df[value_col], errors="coerce")
    m = X.notna() & Y.notna()

    if m.sum() < 10:
        logging.info("Not enough valid rows for violin plot; skipping.")
        return

    # Compute quantile breakpoints over valid data
    qs = X[m].quantile(list(q_edges)).values
    # Ensure strictly increasing cut edges; if duplicates exist, perturb slightly
    edges = np.array(qs, dtype=float)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-9

    labels = [f"Q{i+1}\n[{edges[i]:.2g},{edges[i+1]:.2g}]" for i in range(len(edges) - 1)]
    bins = pd.cut(X[m], bins=edges, labels=labels, include_lowest=True, right=True)
    # Collect lists of values per bin (for violinplot)
    data = [Y[m][bins == lab].values for lab in labels if (bins == lab).any()]

    if not data:
        logging.info("No data per quantile bin for violin plot; skipping.")
        return

    plt.figure(figsize=(7.8, 4.8))
    plt.violinplot(data, showmeans=True, showmedians=False, showextrema=True)
    plt.xticks(ticks=np.arange(1, len(data) + 1), labels=labels, rotation=0)
    plt.title(title or f"{value_col} by {bin_on_col} quantiles")
    plt.xlabel(xlabel or f"{bin_on_col} quantile bins")
    plt.ylabel(ylabel or value_col)
    plt.tight_layout()


# ------------------------------------
# ------------------------------------
# ------- Deprecated functions -------
# ------------------------------------
# ------------------------------------


def plot_histogram_deprecated (rl_copp_obj, model_index, mode, n_bins = 21, title = "Histogram for "):
    """
    Plots a histogram to visualize model behavior metrics such as speed distributions.

    Args:
        rl_copp_obj (RLCoppeliaManager): Manager object holding paths and CLI arguments.
        model_index (int): Index of the model in the list to be analyzed.
        mode (str): Type of data to plot. Currently supports "speeds".
        n_bins (int): Number of bins for the histogram.
        title (str): Prefix for the plot title.
    """
    
    # Get the training csv path for later getting the action times from there
    training_metrics_path = rl_copp_obj.paths["training_metrics"]
    train_records_csv_name = os.path.join(training_metrics_path,"train_records.csv")    # Name of the train records csv to search the algorithm used
    hist_data = []

    if mode == "speeds":
        # Build path to the CSV file containing speed data from testing
        model_id = rl_copp_obj.args.model_ids[model_index]
        robot = rl_copp_obj.args.robot_name
        file_pattern = f"{robot}_model_{model_id}_*_otherdata_*.csv"
        subfolder_pattern = f"{robot}_model_{model_id}_*_testing"
        files = glob.glob(os.path.join(rl_copp_obj.base_path, "robots", robot, "testing_metrics", subfolder_pattern, file_pattern))
        
        if not files:
            logging.error(f"No testing data files found for model index {model_index}.")
            raise FileNotFoundError(f"No testing data files found for model index {model_index} in {os.path.join(rl_copp_obj.base_path, 'robots', robot, 'testing_metrics', subfolder_pattern)}")
        
        # Read CSV
        df = pd.read_csv(files[0])
        data_keys = ['Angular speed', 'Linear speed']
        data_keys_units = ["rad/s", "m/s"]
        bin_min = [-0.5, 0.1]
        bin_max = [0.5, 0.5]
    else:
        logging.error(f"Specified graphs mode doesn't exist: {mode}")
        raise ValueError(f"Invalid mode specified: {mode}")

    for key in data_keys:
        hist_data.append(df[key])

    for i in range(len(data_keys)):
        logging.debug(f"{data_keys[i]} stats:")
        logging.debug(f"Mean: {hist_data[i].mean():.4f}")
        logging.debug(f"Median: {hist_data[i].median():.4f}")
        logging.debug(f"Standard deviation: {hist_data[i].std():.4f}")
        logging.debug(f"Min: {hist_data[i].min():.4f}")
        logging.debug(f"Max: {hist_data[i].max():.4f}")

        # Get timestep to include in plot title
        model_name = f"{robot}_model_{model_id}"
        timestep = (get_data_from_training_csv(model_name, train_records_csv_name, column_header="Action time (s)"))

        # Configure the histogram
        plt.figure(figsize=(10, 6))

        # Create bin equally spaced between the specified limits
        bins = np.linspace(bin_min[i], bin_max[i], n_bins)  # 21 bins for having 20 intervals

        # Create the histogram
        plt.hist(hist_data[i], bins=bins, color='skyblue', edgecolor='black', alpha=0.7)

        # Plot configuration
        plt.title(title + data_keys[i] + ": Model " + str(timestep) + "s", fontsize=14)
        plt.xlabel(f"{data_keys[i]} ({data_keys_units[i]})", fontsize=12)
        plt.ylabel('Frequence', fontsize=12)
        plt.grid(axis='y', alpha=0.75)
        plt.xlim(bin_min[i]-0.05, bin_max[i]+0.05)
        # plt.legend()        

        # Show the histogram
        plt.tight_layout()
        plt.show()


def plot_bars_deprecated(rl_copp_obj, model_index, mode, title="Target Zone Distribution: "):
    """
    Creates a bar chart showing the frequency distribution of discrete values (e.g., target zones).

    Args:
        rl_copp_obj (RLCoppeliaManager): Object containing base paths and CLI arguments.
        model_index (int): Index of the model to analyze.
        mode (str): Type of data to plot (e.g., "target_zones").
        title (str): Title prefix for the chart.
    """
    model_id = rl_copp_obj.args.model_ids[model_index]
    robot_name = rl_copp_obj.args.robot_name

    # Get CSV path
    file_pattern = f"{robot_name}_model_{model_id}_*_test_*.csv"
    subfolder_pattern = f"{robot_name}_model_{model_id}_*_testing"
    files = glob.glob(os.path.join(rl_copp_obj.base_path, "robots", robot_name, "testing_metrics", subfolder_pattern, file_pattern))
    
    if not files:
        logging.error(f"No testing data files found for model index {model_index}.")
        raise FileNotFoundError(f"No testing data files found for model index {model_index} in {os.path.join(rl_copp_obj.base_path, 'robots', robot_name, 'testing_metrics', subfolder_pattern)}")
        

    # Read CSV file
    df = pd.read_csv(files[0])

    if mode != "target_zones":
        logging.error(f"Unsupported bar plot mode: {mode}")
        raise ValueError(f"Invalid mode specified: {mode}")
    
    data_keys = ['Target zone']
    possible_values = [1, 2, 3]
    labels = ['Target zone 1', 'Target zone 2', 'Target zone 3']

    # For each key (although right now the function only works for 'Target zone')
    for key in data_keys:
        data = []
        data = (df[key])
    
        # Count all the samples and calculate percentages
        counts = data.value_counts().reindex(possible_values, fill_value=0)
        total_episodes = len(data)
        percentages = (counts / total_episodes) * 100
        
        # Log statistics
        logging.debug(f"{key} stats:")
        logging.debug(f"Total episodes: {total_episodes}")
        for j in possible_values:
            count = counts.get(j, 0)
            percentage = percentages.get(j, 0)
            logging.debug(f"Zone {j}: {count} episodes ({percentage:.2f}%)")
        
        # Get timestep value of the selected model
        train_records_csv_name = os.path.join(rl_copp_obj.paths["training_metrics"], "train_records.csv")
        model_name = f"{robot_name}_model_{model_id}"
        timestep = get_data_from_training_csv(model_name, train_records_csv_name, column_header="Action time (s)")
        
        # Create the figure
        plt.figure(figsize=(10, 6))      
        
        # Create bars graph
        bars = plt.bar(labels, counts, color=['skyblue', 'lightgreen', 'salmon'], 
                    edgecolor='black', alpha=0.7)
        
        # Add labels
        for bar, count, percentage in zip(bars, counts, percentages):
            plt.text(
                bar.get_x() + bar.get_width()/2, 
                bar.get_height() + 0.5,
                f'{count}\n({percentage:.1f}%)',
                ha='center', va='bottom'
            )
        
        # Plot configuration
        plt.title(f"{title}Model {timestep}s", fontsize=14)
        plt.xlabel('Target Zone', fontsize=12)
        plt.ylabel('Frequence (number of episodes)', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        max_count = counts.max()
        plt.ylim(0, max_count * 1.15)  # 15% aditional space
        plt.tight_layout()

        # Save or show
        if rl_copp_obj.args.save_plots:
            filename = f"bars_{key}.png"
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()


def plot_scene_trajs_with_variability_deprecated(rl_copp_obj, folder_path, num_points=100, nsig=1.0):
    """
    Plot a scene with interpolated mean trajectories from multiple models,
    including uncertainty ellipses at each point based on covariance across trajectories.

    Args:
        rl_copp_obj: Main RL object providing access to config and paths.
        folder_path (str): Path to the folder containing scene and trajectory CSVs.
        num_points (int): Number of interpolation points per trajectory.
        nsig (float): Number of standard deviations for the uncertainty ellipses (e.g., 1.0 or 2.0).
    """
    files = os.listdir(folder_path)
    scene_file = [f for f in files if f.startswith("scene_") and f.endswith(".csv")][0]
    traj_files = [f for f in files if f.startswith("trajectory_") and f.endswith(".csv")]

    logging.debug(f"scene files: {scene_file}")
    logging.debug(f"traj_files: {traj_files}")
    scene_df = pd.read_csv(os.path.join(folder_path, scene_file))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(2.5, -2.5)
    ax.set_ylim(2.5, -2.5)
    ax.set_aspect('equal')
    ax.set_title(f"Scene with Interpolated Mean Trajectories ({nsig}σ uncertainty)")

    # Draw 0.5 m grid
    for i in np.arange(-2.5, 3, 0.5):
        ax.axhline(i, color='lightgray', linewidth=0.5, zorder=0)
        ax.axvline(i, color='lightgray', linewidth=0.5, zorder=0)

    # Draw static scene elements
    for _, row in scene_df.iterrows():
        x, y = row['x'], row['y']
        if row['type'] == 'robot':
            ax.add_patch(plt.Circle((x, y), 0.35 / 2, color='blue', label='Robot', zorder=2))
            
            # Draw orientation
            if 'theta' in row:
                theta = row['theta']

                # Triangle
                front_length = 0.15
                side_offset = 0.08

                # Front point
                front = (x + front_length * np.cos(theta), y + front_length * np.sin(theta))
                # Side points
                left = (x + side_offset * np.cos(theta + 2.5), y + side_offset * np.sin(theta + 2.5))
                right = (x + side_offset * np.cos(theta - 2.5), y + side_offset * np.sin(theta - 2.5))

                triangle = plt.Polygon([front, left, right], color='white', zorder=3)
                ax.add_patch(triangle)
                
        elif row['type'] == 'obstacle':
            ax.add_patch(plt.Circle((x, y), 0.25 / 2, color='gray', label='Obstacle'))
        elif row['type'] == 'target':
            for r, c in [(0.25, 'blue'), (0.125, 'red'), (0.015, 'yellow')]:
                ax.add_patch(plt.Circle((x, y), r, color=c, alpha=0.6))

    # Group trajectories by model ID
    model_trajs = defaultdict(list)
    for file in traj_files:
        parts = file.split('_')
        model_id = parts[-1].split('.')[0]
        model_trajs[model_id].append(os.path.join(folder_path, file))

    colors = plt.cm.get_cmap("tab10")
    training_metrics_path = rl_copp_obj.paths["training_metrics"]
    train_records_csv_name = os.path.join(training_metrics_path, "train_records.csv")

    model_plot_data = []

    # Interpolate and store data for later ordered plotting
    for i, (model_id, paths) in enumerate(model_trajs.items()):
        interpolated_xs, interpolated_ys = [], []
        for path in paths:
            df = pd.read_csv(path)
            x_interp, y_interp = interpolate_trajectory(df['x'].values, df['y'].values, num_points)
            interpolated_xs.append(x_interp)
            interpolated_ys.append(y_interp)

        interpolated_xs = np.array(interpolated_xs)
        interpolated_ys = np.array(interpolated_ys)
        
        mean_x = np.mean(interpolated_xs, axis=0)
        mean_y = np.mean(interpolated_ys, axis=0)
        color = colors((i + 1) % 10)

        model_name = rl_copp_obj.args.robot_name + "_model_" + str(model_id)
        timestep = float(get_data_from_training_csv(model_name, train_records_csv_name, column_header="Action time (s)"))

        model_plot_data.append({
            "timestep": timestep,
            "mean_x": mean_x,
            "mean_y": mean_y,
            "interpolated_xs": interpolated_xs,
            "interpolated_ys": interpolated_ys,
            "color": color,
            "label": f"Model {timestep}s"
        })

    # Sort models by timestep before plotting
    model_plot_data.sort(key=lambda d: d["timestep"])

    for data in model_plot_data:
        ax.plot(data["mean_x"], data["mean_y"], color=data["color"],
                label=data["label"], linewidth=2, zorder=3)

        # for j in range(num_points):
        #     if data["interpolated_xs"].shape[0] < 2:
        #         continue  # Cannot compute covariance with less than 2 trajectories
        #     # point_samples = np.stack((interpolated_xs[:, j], interpolated_ys[:, j]), axis=1)
        #     # draw_robust_uncertainty_ellipse(
        #     #     ax, mean_x[j], mean_y[j], point_samples,
        #     #     color=data["color"], alpha=0.3, nsig=nsig
        #     # )
        #     cov = np.cov(data["interpolated_xs"][:, j], data["interpolated_ys"][:, j])
        #     if not np.isnan(cov).any() and not np.isinf(cov).any():
        #         draw_uncertainty_ellipse(ax, data["mean_x"][j], data["mean_y"][j], cov,
        #                              color=data["color"], nsig=nsig)

    # Deduplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc='upper right')

    plt.grid(True)
    plt.show()


def interpolate_trajectory(x, y, num_points=100):
    """
    Interpolates a trajectory to generate a specified number of evenly spaced points.

    This function takes the x and y coordinates of a trajectory and interpolates them 
    to produce a new trajectory with a uniform distribution of points along its length.

    Args:
        x (array-like): X-coordinates of the original trajectory.
        y (array-like): Y-coordinates of the original trajectory.
        num_points (int): Number of points for the interpolated trajectory.

    Returns:
        tuple:
            - interp_x (array): Interpolated X-coordinates.
            - interp_y (array): Interpolated Y-coordinates.
    """
    distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    cumulative_dist = np.insert(np.cumsum(distances), 0, 0)
    total_length = cumulative_dist[-1]
    if total_length == 0:
        return np.full(num_points, x[0]), np.full(num_points, y[0])

    normalized_dist = cumulative_dist / total_length
    interp_x = interp1d(normalized_dist, x, kind='linear')
    interp_y = interp1d(normalized_dist, y, kind='linear')
    uniform_points = np.linspace(0, 1, num_points)
    return interp_x(uniform_points), interp_y(uniform_points)


def draw_uncertainty_ellipse(ax, mean_x, mean_y, cov, color, nsig=1.0, alpha=0.3, zorder=2):
    """Draws an uncertainty ellipse based on a 2x2 covariance matrix.

    The ellipse represents a confidence region for a 2D Gaussian distribution.

    Args:
        ax (matplotlib.axes.Axes): The axes object to draw the ellipse on.
        mean_x (float): X-coordinate of the ellipse center.
        mean_y (float): Y-coordinate of the ellipse center.
        cov (ndarray): 2x2 covariance matrix.
        color (str or tuple): Color of the ellipse.
        nsig (float, optional): Number of standard deviations for the ellipse size. Defaults to 1.0.
        alpha (float, optional): Transparency of the ellipse (0-1). Defaults to 0.3.
        zorder (int, optional): Drawing order (higher means drawn on top). Defaults to 2.

    Returns:
        None: The ellipse is added to the provided axes object.
    """
    # Compute eigenvalues and eigenvectors of covariance matrix
    vals, vecs = np.linalg.eigh(cov)
    
    # Sort eigenvalues and eigenvectors in descending order
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    
    # Calculate rotation angle (in degrees) from eigenvectors
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    
    # Calculate width and height of ellipse (2 * nsig * standard deviation)
    width, height = 2 * nsig * np.sqrt(vals)

    # Create ellipse patch
    ell = Ellipse(
        xy=(mean_x, mean_y),     # Ellipse center
        width=width,             # Major axis length
        height=height,           # Minor axis length
        angle=theta,             # Rotation angle in degrees
        color=color,             # Color
        alpha=alpha,             # Transparency
        zorder=zorder            # Drawing order
    )
    ax.add_patch(ell)


def draw_robust_uncertainty_ellipse(ax, mean_x, mean_y, points, color='gray', alpha=0.3, zorder=1, nsig=2.0):
    """
    Draws a robust uncertainty ellipse based on a set of 2D points.

    This function computes a robust covariance matrix using the Minimum Covariance Determinant (MCD) 
    estimator and uses it to draw an uncertainty ellipse. If the robust estimation fails, it falls 
    back to the classical covariance matrix.

    Args:
        ax (matplotlib.axes.Axes): The axes object to draw the ellipse on.
        mean_x (float): X-coordinate of the ellipse center.
        mean_y (float): Y-coordinate of the ellipse center.
        points (ndarray): Array of shape (n_samples, 2) containing the 2D points.
        color (str or tuple): Color of the ellipse.
        alpha (float, optional): Transparency of the ellipse (0-1). Defaults to 0.3.
        zorder (int, optional): Drawing order (higher means drawn on top). Defaults to 1.
        nsig (float, optional): Number of standard deviations for the ellipse size. Defaults to 2.0.

    Returns:
        None: The ellipse is added to the provided axes object.
    """
    if len(points) < 2:
        return

    try:
        # Primer intento: robusto
        robust_cov = MinCovDet(support_fraction=0.9).fit(points)
        cov = robust_cov.covariance_
    except Exception as e:
        logging.warning(f"[MCD fallback] Using classical covariance due to error: {e}")
        try:
            cov = np.cov(points.T)
        except Exception as e2:
            logging.warning(f"Failed to compute classical covariance too: {e2}")
            return

    try:
        # Descomposición de la matriz de covarianza
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals = vals[order]
        vecs = vecs[:, order]

        # Parámetros de la elipse
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 2 * nsig * np.sqrt(vals)

        ell = Ellipse(
            xy=(mean_x, mean_y),
            width=width,
            height=height,
            angle=theta,
            color=color,
            alpha=alpha,
            zorder=zorder
        )
        ax.add_patch(ell)
    except Exception as e:
        logging.warning(f"Failed to draw ellipse: {e}")


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from PIL import Image
from scipy.ndimage import distance_transform_edt

# ----------------------------- utilities -----------------------------

def _img_to_gray_uint8(img_path: str) -> np.ndarray:
    """
    Load an image and return a grayscale uint8 array of shape (H, W),
    with values in [0, 255].
    """
    img = Image.open(img_path).convert("L")  # 8-bit grayscale
    return np.array(img, dtype=np.uint8)

def _world_to_pixel(x: float, y: float, *, m_per_px: float, origin_xy: tuple[float,float],
                    h_px: int, origin_is_lower_left: bool) -> tuple[int, int]:
    """
    Convert world (meters) to pixel (row, col). Returns (py, px) integer indices.
    """
    x0, y0 = origin_xy
    px = int(round((x - x0) / m_per_px))
    py = int(round((y - y0) / m_per_px))
    if not origin_is_lower_left:
        py = (h_px - 1) - py
    return py, px

def _pixel_to_world(px: int, py: int, *, m_per_px: float, origin_xy: tuple[float,float],
                    h_px: int, origin_is_lower_left: bool) -> tuple[float, float]:
    """
    Convert pixel (row, col) to world (meters). Input is (py, px).
    """
    x0, y0 = origin_xy
    if not origin_is_lower_left:
        py = (h_px - 1) - py
    x = x0 + px * m_per_px
    y = y0 + py * m_per_px
    return float(x), float(y)

# ----------------------------- 1) occupancy mask -----------------------------

def build_occupancy_mask_from_png(
    map_png_path: str,
    *,
    obstacle_threshold: int = 15,
    m_per_px: float = 0.02013,
    clearance_m: float = 0.35,
) -> dict:
    """
    Build an occupancy mask from a PNG occupancy-like map.

    Pixels considered obstacle if gray ∈ [0, obstacle_threshold] (inclusive).
    Any other gray (e.g., light gray/white) is considered free.

    The routine also inflates the obstacles by a given clearance (meters)
    using a distance transform, so invalid pixels include both the solid
    obstacles and any pixel closer than `clearance_m`.

    Parameters
    ----------
    map_png_path : str
        Path to the map image (PNG).
    obstacle_threshold : int, default=15
        Gray threshold (0..255). Pixels ≤ threshold are obstacles.
    m_per_px : float, default=0.02013
        Map resolution in meters per pixel.
    clearance_m : float, default=0.35
        Safety margin around obstacles in meters.

    Returns
    -------
    result : dict
        {
          "occ_raw": np.ndarray bool, shape (H,W)  # True = obstacle pixel
          "occ_inflated": np.ndarray bool, shape (H,W)  # True = invalid (obstacle or too close)
          "dist_to_obstacle_m": np.ndarray float, shape (H,W)  # distance to nearest obstacle [m]
          "size": (H, W),
          "m_per_px": float
        }
    """
    gray = _img_to_gray_uint8(map_png_path)
    h, w = gray.shape

    # raw obstacle mask: black/dark (0..threshold) = True
    occ_raw = (gray <= obstacle_threshold)

    # distance (in pixels) from each pixel to nearest obstacle.
    # edt computes distance to the *zero* pixels, so use ~occ_raw:
    dist_px = distance_transform_edt(~occ_raw)

    dist_m = dist_px * m_per_px
    occ_inflated = occ_raw | (dist_m < clearance_m)

    return {
        "occ_raw": occ_raw,
        "occ_inflated": occ_inflated,
        "dist_to_obstacle_m": dist_m,
        "size": (h, w),
        "m_per_px": m_per_px,
    }


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

# TODO close_tol is not working properly
def interactive_polygon_on_map_live(
    map_png_path: str,
    *,
    m_per_px: float,
    origin_xy: tuple[float, float],
    origin_is_lower_left: bool = False,
    close_tol: float = 0.3,  # meters: click near the first vertex to auto-finish
    title: str = ("Click vertices to draw polygon. "
                  "Click near the first to close. Enter=finish, Backspace/U=undo, Esc=cancel")
) -> np.ndarray:
    """
    Interactive polygon over a map in WORLD coordinates, with live edges.
    - Left-click adds a vertex.
    - Clicking within `close_tol` meters of the FIRST vertex closes the polygon and finishes.
    - Press Enter/Return to finish (if there are >=3 points, it will close last->first).
    - Backspace or 'U' undoes the last point.
    - Esc cancels (returns empty array).
    Returns (N,2) vertices in meters (unique; first is NOT repeated at the end).
    """
    # --- background image in world coords
    img = Image.open(map_png_path).convert("RGB")
    w_px, h_px = img.size
    x0, y0 = origin_xy
    x1 = x0 + w_px * m_per_px
    y1 = y0 + h_px * m_per_px
    origin_kw = "lower" if origin_is_lower_left else "upper"

    fig, ax = plt.subplots(figsize=(10, 8), dpi=120)
    ax.imshow(img, extent=[x0, x1, y0, y1], origin=origin_kw)
    ax.set_aspect("equal")
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
    ax.set_title(title)

    # state
    pts: list[tuple[float, float]] = []
    markers = []
    line = Line2D([], [], color="cyan", lw=2)
    ax.add_line(line)
    finished = False
    cancelled = False

    def _redraw():
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        line.set_data(xs, ys)
        fig.canvas.draw_idle()

    def _cleanup_close():
        """Disconnect callbacks and close the figure safely."""
        fig.canvas.mpl_disconnect(cid_c)
        fig.canvas.mpl_disconnect(cid_k)
        plt.close(fig)

    def on_click(event):
        nonlocal finished
        if finished or event.inaxes != ax or event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return  # clicked outside axes area

        x, y = float(event.xdata), float(event.ydata)

        # Close if near the first vertex (meters)
        if len(pts) >= 3:
            d = np.hypot(x - pts[0][0], y - pts[0][1])
            if d <= close_tol:
                # visually show closing edge
                pts.append(pts[0])
                _redraw()
                # drop duplicate so returned polygon has unique vertices
                pts.pop()
                finished = True
                _cleanup_close()
                return

        # Add vertex
        pts.append((x, y))
        m = ax.plot([x], [y], marker="o", color="cyan", ms=6)[0]
        markers.append(m)
        _redraw()

    def on_key(event):
        nonlocal finished, cancelled
        key = (event.key or "").lower()
        if key in ("escape", "esc"):
            cancelled = True
            _cleanup_close()
        elif key in ("enter", "return"):
            # finish if we have a valid polygon
            if len(pts) >= 3 and not finished:
                # close by connecting last->first for consistency
                pts.append(pts[0])
                _redraw()
                pts.pop()
                finished = True
            _cleanup_close()
        elif key in ("backspace", "u"):
            if finished or not pts:
                return
            pts.pop()
            if markers:
                mk = markers.pop()
                mk.remove()
            _redraw()

    cid_c = fig.canvas.mpl_connect("button_press_event", on_click)
    cid_k = fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()

    # finalize
    if cancelled or len(pts) < 3:
        return np.empty((0, 2), dtype=float)
    return np.asarray(pts, dtype=float)



def interactive_polygon_on_map(
    map_png_path: str,
    *,
    m_per_px: float,
    origin_xy: tuple[float,float],
    origin_is_lower_left: bool = False,
    title: str = "Click polygon vertices; Enter to finish. Click near start to close.",
) -> np.ndarray:
    """
    Let the user draw a polygon directly on the map in WORLD coordinates.

    - Displays the PNG placed at (origin_xy, m_per_px) with the correct y-origin.
    - User clicks vertices in order; press Enter to finish.
    - If the last click is close to the first, the polygon is closed automatically.

    Returns
    -------
    poly_xy : np.ndarray, shape (M, 2)
        Vertices (x,y) in meters. If the user cancels (no clicks), returns an empty array.
    """
    img = Image.open(map_png_path).convert("RGB")
    w, h = img.size
    x0, y0 = origin_xy
    x1 = x0 + w * m_per_px
    y1 = y0 + h * m_per_px
    origin_kw = "lower" if origin_is_lower_left else "upper"

    fig, ax = plt.subplots(figsize=(10, 8), dpi=120)
    ax.imshow(img, extent=[x0, x1, y0, y1], origin=origin_kw)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")

    pts = plt.ginput(n=-1, timeout=0)  # arbitrary number of clicks, Enter to end
    plt.close(fig)

    if len(pts) == 0:
        return np.empty((0,2), dtype=float)

    poly = np.array(pts, dtype=float)

    # If last vertex is close to the first, drop it and close polygon logically
    if len(poly) >= 3 and np.linalg.norm(poly[-1] - poly[0]) < 0.02:  # 2 cm tolerance
        poly = poly[:-1]

    return poly

# ----------------------------- 3) grid of valid positions -----------------------------

def grid_positions_from_mask(
    *,
    occ_inflated: np.ndarray,
    map_png_path: str,
    m_per_px: float,
    origin_xy: tuple[float,float],
    origin_is_lower_left: bool = False,
    grid_step_m: float = 0.25,
    polygon_xy: np.ndarray | None = None,
) -> np.ndarray:
    """
    Produce a list of valid (x,y) world positions sampled on a grid.

    A position is valid if:
      - It lies INSIDE the polygon of interest (if provided).
      - It falls INSIDE the image bounds.
      - The corresponding pixel is NOT invalid (occ_inflated == False).

    Parameters
    ----------
    occ_inflated : np.ndarray bool, shape (H,W)
        True for invalid pixels (obstacle or too close). Built by `build_occupancy_mask_from_png`.
    map_png_path : str
        Path to the PNG (used only for size/meta safety).
    m_per_px : float
        Meters per pixel.
    origin_xy : tuple[float,float]
        (x_min, y_min) where the image is placed in world coordinates.
    origin_is_lower_left : bool, default=False
        If True, the image y-origin is bottom-left; otherwise top-left.
    grid_step_m : float, default=0.25
        Grid step in meters.
    polygon_xy : np.ndarray or None
        Optional polygon (M,2) in world coords. If None, we accept the whole image rectangle.

    Returns
    -------
    positions_xy : np.ndarray, shape (K, 2)
        Valid (x,y) positions in meters.
    """
    h, w = occ_inflated.shape
    x0, y0 = origin_xy
    x1 = x0 + w * m_per_px
    y1 = y0 + h * m_per_px

    # grid over the whole image rect (then clip to polygon)
    xs = np.arange(x0, x1 + 1e-9, grid_step_m)
    ys = np.arange(y0, y1 + 1e-9, grid_step_m)
    X, Y = np.meshgrid(xs, ys)
    candidates = np.column_stack([X.ravel(), Y.ravel()])

    # polygon filter (if provided)
    if polygon_xy is not None and len(polygon_xy) >= 3:
        poly_path = MplPath(polygon_xy, closed=True)
        inside_poly = poly_path.contains_points(candidates)
        candidates = candidates[inside_poly]

    # keep only those inside image bounds & free in inflated mask
    valid = []
    for (x, y) in candidates:
        py, px = _world_to_pixel(x, y, m_per_px=m_per_px, origin_xy=origin_xy,
                                 h_px=h, origin_is_lower_left=origin_is_lower_left)
        if 0 <= px < w and 0 <= py < h and not occ_inflated[py, px]:
            valid.append((x, y))

    return np.array(valid, dtype=float)

# ----------------------------- convenience wrapper -----------------------------

def build_valid_positions_from_map(
    map_png_path: str,
    *,
    m_per_px: float = 0.02013,
    origin_xy: tuple[float,float] = (-10.5, -6.0),
    origin_is_lower_left: bool = False,
    obstacle_threshold: int = 15,
    clearance_m: float = 0.35,
    grid_step_m: float = 0.25,
    interactive_polygon: bool = True,
) -> dict:
    """
    High-level helper that:
      1) Builds an inflated occupancy mask from the PNG.
      2) Lets the user draw a polygon of interest (optional).
      3) Samples a grid of valid (x,y) positions within the polygon and away from obstacles.

    Returns
    -------
    out : dict
        {
          "occ_mask": occ_inflated (H,W) bool,
          "dist_to_obstacle_m": dist_m (H,W) float,
          "polygon_xy": np.ndarray (M,2) or empty array,
          "positions_xy": np.ndarray (K,2) world positions,
          "meta": {"m_per_px":..., "origin_xy":..., "origin_is_lower_left":..., "size":(H,W)}
        }
    """
    mask_data = build_occupancy_mask_from_png(
        map_png_path,
        obstacle_threshold=obstacle_threshold,
        m_per_px=m_per_px,
        clearance_m=clearance_m
    )
    occ_infl = mask_data["occ_inflated"]
    h, w = mask_data["size"]

    if interactive_polygon:
        poly = interactive_polygon_on_map_live(
            map_png_path,
            m_per_px=m_per_px,
            origin_xy=origin_xy,
            origin_is_lower_left=origin_is_lower_left,
            close_tol = 0.05,
            title="Click polygon vertices; Enter to finish. (Close by clicking near start)"
        )
    else:
        poly = np.empty((0,2), dtype=float)

    positions = grid_positions_from_mask(
        occ_inflated=occ_infl,
        map_png_path=map_png_path,
        m_per_px=m_per_px,
        origin_xy=origin_xy,
        origin_is_lower_left=origin_is_lower_left,
        grid_step_m=grid_step_m,
        polygon_xy=poly if len(poly) >= 3 else None
    )

    return {
        "occ_mask": occ_infl,
        "dist_to_obstacle_m": mask_data["dist_to_obstacle_m"],
        "polygon_xy": poly,
        "positions_xy": positions,
        "meta": {
            "m_per_px": m_per_px,
            "origin_xy": origin_xy,
            "origin_is_lower_left": origin_is_lower_left,
            "size": (h, w),
        }
    }

# ----------------------------- (optional) quick visual check -----------------------------

def preview_mask_and_positions(
    map_png_path: str,
    result: dict,
):
    """
    Quick visual check: show map, inflated occupancy mask overlay, polygon (if any),
    and sampled valid positions as dots.
    """
    m_per_px = result["meta"]["m_per_px"]
    origin_xy = result["meta"]["origin_xy"]
    origin_is_lower_left = result["meta"]["origin_is_lower_left"]
    occ = result["occ_mask"]
    poly = result["polygon_xy"]
    pts = result["positions_xy"]

    img = Image.open(map_png_path).convert("RGB")
    w, h = img.size
    x0, y0 = origin_xy
    x1 = x0 + w * m_per_px
    y1 = y0 + h * m_per_px
    origin_kw = "lower" if origin_is_lower_left else "upper"

    fig, ax = plt.subplots(figsize=(10,8), dpi=120)
    ax.imshow(img, extent=[x0,x1,y0,y1], origin=origin_kw)
    # show inflated mask in red translucency
    ax.imshow(occ.astype(float), extent=[x0,x1,y0,y1], origin=origin_kw,
              cmap="Reds", alpha=0.35, vmin=0, vmax=1)
    if len(poly) >= 3:
        ax.plot(*poly.T, "-c", lw=2)
        ax.plot([poly[-1,0], poly[0,0]], [poly[-1,1], poly[0,1]], "-c", lw=2)
    if pts.size:
        ax.scatter(pts[:,0], pts[:,1], s=12, c="lime", edgecolors="k", linewidths=0.3)
    ax.set_aspect("equal"); ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
    ax.set_title("Inflated occupancy (red) + valid grid positions (green)")
    fig.tight_layout()
    plt.show()





def extract_timestamp(path: str) -> str:
    """
    Get the timestamp from a CSV filename.
    ..._YYYY-MM-DD_HH-MM-SS.csv
    """
    m = re.search(r'_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.csv$', os.path.basename(path))
    return m.group(1) if m else None



def _yes(prompt: str, default=False) -> bool:
    '''
    Ask a yes/no question via input() and return their answer as a boolean.
    Args:
        prompt (str): The question to ask the user.
        default (bool): The default answer if the user just hits Enter. Defaults to False.
    Returns:
        bool: True for 'yes', False for 'no'.
    '''
    s = input(f"{prompt} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
    if s == "" and default: 
        return True
    return s in ("y", "yes", "s", "si", "sí")


def _ask_float(prompt: str, current: float):
    '''
    Ask the user to input a float value, with the option to keep the current value.
    Args:
        prompt (str): The prompt to display to the user.
        current (float): The current value to keep if the user presses Enter.
    Returns:
        float | None: The new float value or the current value if Enter is pressed.
    '''
    s = input(f"{prompt} - Enter to keep current value ({current:.3f}): ").strip()
    if s == "":
        return current
    try:
        return float(s)
    except ValueError:
        print("  [!] Invalid value, current will be kept.")
        return current


def ask_and_set_limits_for_axes(ax, label: str):
    '''
    Interactively ask the user if they want to set custom axis limits for a matplotlib Axes object.
    If the user agrees, prompts for min and max values for both X and Y axes.
    Args:
        ax (matplotlib.axes.Axes): The Axes object to modify.
        label (str): Label to identify the graph in prompts.
    '''
    print(f"\n--- Limits for «{label}» ---")
    if not _yes("Do you want to define custom axis limits for this graph?"):
        return  

    # Current limit values
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    # X axis
    if _yes("Fix limit in X axis?"):
        nx0 = _ask_float("  xmin", x0)
        nx1 = _ask_float("  xmax", x1)
        if nx1 <= nx0:
            print("  [!] xmax must be > xmin. Limits in X will be ignored.")
        else:
            ax.set_xlim(nx0, nx1)

    # Y axis
    if _yes("Fix limits in Y?"):
        ny0 = _ask_float("  ymin", y0)
        ny1 = _ask_float("  ymax", y1)
        if ny1 <= ny0:
            print("  [!] ymax debe ser > ymin. Limits in Y will be ignored.")
        else:
            ax.set_ylim(ny0, ny1)


def map_files_by_timestamp(files_1, files_2=None):
    '''
    Maps files by their extracted timestamps.
    If files2 is provided, returns only timestamps present in both lists.
    Args:
        files_1 (list): List of file paths.
        files_2 (list, optional): Second list of file paths for intersection. Defaults to None.
    Returns:
        tuple: (path_1, path_2) where path_1 is from files_1 and path_2 from files_2 (or None if files_2 is None).
    '''

    # If only one timestamp, return it directly
    if len(files_1) == 1:
        if files_2 is not None:
            return files_1[0], files_2[0]
        else:
            return files_1[0], None

    # Map timestamps to file paths 1
    timestamp_1= {}
    for p in files_1:
        ts = extract_timestamp(p)
        if ts:
            timestamp_1[ts] = p
    

    if files_2 is not None:
        timestamp_2 = {}
        for p in files_2:
            ts = extract_timestamp(p)
            if ts:
                timestamp_2[ts] = p

        # Check for common timestamps
        common_ts = sorted(set(timestamp_1.keys()).intersection(timestamp_2.keys()))
        if not common_ts:
            logging.error("[!] No matching timestamps.")
            logging.info(f"Folder 1: {sorted(timestamp_1.keys())}")
            logging.info(f"Folder 2: {sorted(timestamp_2.keys())}")
            return
    else:
        common_ts = sorted(timestamp_1.keys())
        
    # Show avaliable timestamps and ask user to choose one
    print("\nAvaliable timestamps:")
    for i, ts in enumerate(common_ts):
        print(f"  [{i}] {ts}")

    while True:
        sel = input("Choose the desired experiment to load: ").strip()
        if sel.isdigit():
            sel = int(sel)
            if 0 <= sel < len(common_ts):
                chosen_ts = common_ts[sel]
                break
        print(f"Invalid index. Please enter a number between 0 and {len(common_ts)-1}.")

    # Rutas seleccionadas
    path_1 = timestamp_1[chosen_ts]
    if files_2 is not None:
        path_2  = timestamp_2[chosen_ts]
    else:
        path_2 = None
    return path_1, path_2


def unwrap_env(vec_env, idx=0):
    """
    Recursively unwrap a vectorized environment to get the base environment.
    Args:
        vec_env: The vectorized environment (VecEnv).
        idx (int): Index of the environment to unwrap (default is 0).
    Returns:
        The unwrapped base environment.
    """
    env = getattr(vec_env, "envs", [vec_env])[idx]
    while hasattr(env, "env"):
        env = env.env
    return env



# ------------------------------------
# ------------------------------------
# ------- CLI Terminal helpers -------
# ------------------------------------
# ------------------------------------


def ensure_ttt(name: str) -> str:
    """Ensure the scene filename ends with .ttt (case-insensitive)."""
    name = name.strip()
    return name if name.lower().endswith(".ttt") else f"{name}.ttt"


def arrow_menu(options: List[str], title: str = "Select an option") -> str:
    """Arrow-key menu using curses. Returns the selected option text.

    Controls:
      - Up/Down arrows to move
      - Enter to select
      - 'q' to quit (returns the last highlighted)
    """
    idx = 0

    def _draw(stdscr):
        nonlocal idx
        curses.curs_set(0)
        stdscr.nodelay(False)
        stdscr.keypad(True)

        while True:
            stdscr.clear()
            maxy, maxx = stdscr.getmaxyx()
            stdscr.addstr(0, 0, title[:maxx-1], curses.A_BOLD)

            start_line = 2
            for i, opt in enumerate(options):
                prefix = "➤ " if i == idx else "  "
                line = f"{prefix}{opt}"
                if i == idx:
                    stdscr.addstr(start_line + i, 0, line[:maxx-1], curses.A_REVERSE)
                else:
                    stdscr.addstr(start_line + i, 0, line[:maxx-1])

            key = stdscr.getch()
            if key in (curses.KEY_UP, ord('k')):
                idx = (idx - 1) % len(options)
            elif key in (curses.KEY_DOWN, ord('j')):
                idx = (idx + 1) % len(options)
            elif key in (curses.KEY_ENTER, 10, 13):
                return
            elif key in (ord('q'), 27):  # q or ESC
                return

    curses.wrapper(_draw)
    return options[idx]


def prompt_str(label: str, default: Optional[str] = None, allow_empty: bool = True) -> str:
    """Ask for a string with Ctrl+B and Ctrl+C support."""
    while True:
        s = _read_line_ctrlb(label, default=default)
        if s == "" and default is not None:
            return default
        if not allow_empty and s.strip() == "":
            print("This field cannot be empty.")
            continue
        return s


def prompt_int(label: str, default: Optional[int] = None,
               min_val: Optional[int] = None, max_val: Optional[int] = None) -> int:
    """Ask for an integer with optional bounds."""
    while True:
        s = _read_line_ctrlb(label, default=None if default is None else str(default))
        if s == "" and default is not None:
            v = default
        else:
            try:
                v = int(s)
            except ValueError:
                print("Invalid integer.")
                continue
        if min_val is not None and v < min_val:
            print(f"Value must be >= {min_val}.")
            continue
        if max_val is not None and v > max_val:
            print(f"Value must be <= {max_val}.")
            continue
        return v


def prompt_float(label: str, default: Optional[float] = None,
                 min_val: Optional[float] = None, max_val: Optional[float] = None) -> float:
    """Ask for a float with optional bounds."""
    while True:
        s = _read_line_ctrlb(label, default=None if default is None else str(default))
        if s == "" and default is not None:
            v = default
        else:
            try:
                v = float(s)
            except ValueError:  # User could haver entered "pi" or "-pi"
                if s.lower() == "pi" or s.lower()=="-pi":
                    v = np.pi if s=="pi" else -np.pi
                else:
                    print("Invalid number.")
                    continue
        if min_val is not None and v < min_val:
            print(f"Value must be >= {min_val}.")
            continue
        if max_val is not None and v > max_val:
            print(f"Value must be <= {max_val}.")
            continue
        return v


def prompt_choice(label: str, options: Iterable[str], default_idx: int = 0) -> str:
    """Ask user to pick an option by index."""
    opts = list(options)
    for i, opt in enumerate(opts):
        print(f"  [{i}] {opt}")
    idx = prompt_int(f"{label} (pick index)", default=default_idx,
                     min_val=0, max_val=len(opts) - 1)
    return opts[idx]


def prompt_confirm(label: str, default: bool = True) -> bool:
    """Yes/No prompt with Ctrl+B and Ctrl+C support."""
    suffix = "Y/n" if default else "y/N"
    ans = prompt_str(f"{label} ({suffix})", default="y" if default else "n").strip().lower()
    return ans in ("y", "yes")

class BackSignal(Exception):
    """Internal signal to go back one step (raised on Ctrl+B)."""
    pass

def _read_line_ctrlb(label: str, default: Optional[str] = None) -> str:
    """Read a line (Windows + POSIX) with:
       - Ctrl+B -> back one step (BackSignal)
       - Ctrl+C -> cancel (KeyboardInterrupt)
       - Clean line (no weird indentation)
    """
    import os, sys
    # Start on a fresh line, move to column 0, clear line
    sys.stdout.write("\n\r\033[2K")
    # Prompt (no leading spaces)
    if default not in (None, ""):
        sys.stdout.write(f"{label} [{default}]: ")
    else:
        sys.stdout.write(f"{label}: ")
    sys.stdout.flush()

    buf: List[str] = []

    if os.name == "nt":
        import msvcrt
        while True:
            ch = msvcrt.getwch()
            if ch in ("\r", "\n"):
                sys.stdout.write("\n"); sys.stdout.flush()
                text = "".join(buf)
                return text if text or default is None else str(default)
            if ch == "\x03":  # Ctrl+C
                sys.stdout.write("\n"); sys.stdout.flush()
                raise KeyboardInterrupt
            if ch == "\x02":  # Ctrl+B
                sys.stdout.write("\n"); sys.stdout.flush()
                raise BackSignal()
            if ch in ("\b", "\x7f"):  # Backspace
                if buf:
                    buf.pop(); sys.stdout.write("\b \b"); sys.stdout.flush()
                continue
            if ch in ("\x00", "\xe0"):  # special keys
                _ = msvcrt.getwch()
                continue
            buf.append(ch); sys.stdout.write(ch); sys.stdout.flush()
    else:
        import termios, tty
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while True:
                ch = sys.stdin.read(1)
                if ch in ("\r", "\n"):
                    sys.stdout.write("\n"); sys.stdout.flush()
                    text = "".join(buf)
                    return text if text or default is None else str(default)
                if ch == "\x03":  # Ctrl+C
                    sys.stdout.write("\n"); sys.stdout.flush()
                    raise KeyboardInterrupt
                if ch == "\x02":  # Ctrl+B
                    sys.stdout.write("\n"); sys.stdout.flush()
                    raise BackSignal()
                if ch in ("\x7f", "\b"):
                    if buf:
                        buf.pop(); sys.stdout.write("\b \b"); sys.stdout.flush()
                    continue
                if ch.isprintable():
                    buf.append(ch); sys.stdout.write(ch); sys.stdout.flush()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)





def print_uncore_logo():
    """Print the UnCoRE logo (ASCII version) with blue color and proper formatting."""
    blue = "\033[94m"   # Bright blue
    reset = "\033[0m"   # Reset color
    logo = r"""
    

                                                                                                                  
                                                                                                  :@@@.           
                                                                                               .*@@@@@@=          
                                                                                             -@@@@@@@@@@.         
                                 .@%==.                                          -=.       -@@@@@@@@@@@=          
                               .@#                                             +@@@@*    .@@@@@@@-.               
                             .%%.                                              #@@@@@   *@@@@@-                   
                           .#@.                                                 =@@+  .@@@@@.                     
                          #@.                                                        :@@@@:        +@@@@@@%:      
  @@=      %@@@@@@@@.   *@.       .@@@@@@@@@   .@@@@@@@@@@@@@@@@#                    @@@@-       =@@@@@@@@@@@.    
   %+      @:     .@:   @-        .@      +@   :@.      .....  =@          .#@@-    -@@@@       :@@@@@@@@@@@@@    
   %+      @:     .@:   @-        .@      +@   :@.     :@====.            :@@@@@+   #@@@=       @@@@@@@@@@@@@@:   
   %+      @:     .@:   @-        .@      +@   :@.     :@                 .@@@@@-   #@@@+       #@@@@@@@@@@@@@.   
   %+      @:     .@:   @-        .@      +@   :@.     :@                    :.     .@@@@       .@@@@@@@@@@@@*    
   %+      @:     .@:   @-        .@      +@   :@.     :@      =@                    %@@@%        *@@@@@@@@@-     
   +%%%%%%%%.     .*%#. #@.       .%%%%%%%%#   .#      .%%%%%%%%*                    .@@@@@         .=#%*:        
                         .@#                                                   .@@@@:  %@@@@*                     
                           -@=                                                 %@@@@@   -@@@@@%.                  
                             =@:                                               :@@@@-     *@@@@@@@+-.             
                               *@                                                           #@@@@@@@@@@@          
                                 %@.                                                          +@@@@@@@@@.         
                                   -=-                                                          .@@@@@@:          
                                                                                                   =%+            
                                                                                                                  


    """
    print(f"{blue}{logo}{reset}")
