from collections import defaultdict
import csv
import datetime
import glob
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import re
import shutil
import subprocess
import sys
import time
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

AGENT_SCRIPT_COPPELIA = "/Agent_Script"
AGENT_SCRIPT_PYTHON = "common/agent_copp.py"


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

    if hasattr(self.args, "robot_name") and not self.args.robot_name:
        logging.warning("WARNING: '--robot_name' was not specified, so default name 'burgerBot' will be used")
    
    if hasattr(self.args, "params_file") and not self.args.params_file:
        if self.args.command == 'train':
            self.args.params_file = os.path.join(self.base_path, "configs", "params_default_file.json")
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
        json_file = os.path.join(rl_copp_obj.paths["parameters_used"], f"params_file_model_{model_id}.json")
        
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


def get_model_names_and_paths(rl_copp_obj):
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
    will be unique, as it makes use of the timestamp. In case of inference, it will generate also the path of the csv 
    to store the speeds of the robot during the testing process.

    Args:
        model_name (str): Name of the model file, as it will be used for identifying the csv file.
        metrics_path (str): Path to store the csv files with the obtained metrics.
        train_flag (bool): True if the script has been executed in training mode, False in case of running a test. True by default.

    Return: #TODO complete the return
        output_csv_path (str): Path to the new csv file.
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
    # Just keep the name until the '_best' part
    if "_best" in model_name:
        model_name = model_name.split("_best")[0]

    with open(csv_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        # Get the key of the first column (assumed to contain the model names)
        first_col = reader.fieldnames[0]
        for row in reader:
            if row[first_col] == model_name:
                return row.get("Algorithm")
            
    logging.error("There was an error while checking the algorithm used for training the model")
    raise ValueError(f"Model '{model_name}' not found in CSV file '{csv_path}'")


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


def update_and_copy_script(rl_copp_obj):
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
    agent_object = rl_copp_obj.current_sim.getObject(AGENT_SCRIPT_COPPELIA)
    agent_script_handle = rl_copp_obj.current_sim.getScript(1, agent_object)

    # Read Agent_script.py  
    agent_script_path = os.path.join(rl_copp_obj.base_path, "src", AGENT_SCRIPT_PYTHON)
    logging.info(f"Copying content of {agent_script_path} inside the scene in {AGENT_SCRIPT_COPPELIA}")

    try:
        with open(agent_script_path, "r") as file:
            script_content = file.read()

        # Dictionary with variables to update
        if not hasattr(rl_copp_obj.args, "model_name"):
            
            rl_copp_obj.args.model_name = None
            model_name = None
        else: 
            try:
                model_name = os.path.basename(rl_copp_obj.args.model_name)
            except:
                model_name = None

        if not hasattr(rl_copp_obj.args, "scene_to_load_folder"):
            scene_to_load_folder = None
        else:
            scene_to_load_folder = rl_copp_obj.args.scene_to_load_folder

        if not hasattr(rl_copp_obj.args, "save_scene"):
            save_scene = None
        else:
            save_scene = rl_copp_obj.args.save_scene

        if not hasattr(rl_copp_obj.args, "save_traj"):
            save_traj = None
        else:
            save_traj = rl_copp_obj.args.save_traj

        if not hasattr(rl_copp_obj.args, "model_ids"):
            model_ids = None
            action_times = None
            amp_model_ids = None
        else:
            model_ids = rl_copp_obj.args.model_ids

            # Get how many targets are in the scene
            scene_configs_path = rl_copp_obj.paths["scene_configs"]
            scene_path = os.path.join(scene_configs_path, rl_copp_obj.args.scene_to_load_folder)
            scene_path_csv = find_scene_csv_in_dir(scene_path)
            if not os.path.exists(scene_path_csv):
                logging.error(f"[ERROR] CSV scene file not found: {scene_path_csv}")
                sys.exit()
            df = pd.read_csv(scene_path_csv)
            num_targets = (df['type'] == 'target').sum()

            # Create a list of model IDs, repeating each ID for the number of iterations specified
            amp_model_ids = [model_id for model_id in model_ids for _ in range(rl_copp_obj.args.iters_per_model)]
            
            # If the number of targets is greater than 1, repeat the model IDs to match the number of targets
            amp_model_ids = amp_model_ids * num_targets

            # Set the model IDs in the args
            rl_copp_obj.args.model_ids = amp_model_ids
            action_times= get_fixed_actimes(rl_copp_obj)
            print(action_times)


        replacements = {
            "robot_name": rl_copp_obj.args.robot_name,
            "model_name": model_name,
            "model_ids": amp_model_ids,
            "base_path": rl_copp_obj.base_path,
            "comms_port": rl_copp_obj.free_comms_port,
            "verbose": rl_copp_obj.args.verbose,
            "scene_to_load_folder": scene_to_load_folder,
            "save_scene": save_scene,
            "save_traj": save_traj,
            "testvar": rl_copp_obj.free_comms_port+1,
            "action_times": action_times,
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
        for key, value in rl_copp_obj.params_env.items():
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
        rl_copp_obj.current_sim.setScriptText(agent_script_handle, script_content)
        logging.info("Script updated successfully in CoppeliaSim.")
        return True

    except Exception as e:
        logging.error(f"Something happened while trying to update the content of the script inside Coppelia's scene: {e}")
        # sys.exit()


def create_discs_under_target(rl_copp_obj):
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
        ("Target_disc_1", [0, 0, 1], 0.002, rl_copp_obj.params_env["reward_dist_1"]),  # Blue
        ("Target_disc_2", [1, 0, 0], 0.004, rl_copp_obj.params_env["reward_dist_2"]),  # Red
        ("Target_disc_3", [1, 1, 0], 0.006, rl_copp_obj.params_env["reward_dist_3"])   # Yellow
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


def start_coppelia_and_simulation(rl_copp_obj):
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
        try:
            rl_copp_obj.args.scene_path = os.path.join(rl_copp_obj.base_path, "scenes", f"{rl_copp_obj.args.robot_name}_scene.ttt")
        except:
            rl_copp_obj.args.scene_path = os.path.join(rl_copp_obj.base_path, "scenes/burgerBot_scene.ttt")

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
                    f"--title={rl_copp_obj.terminal_id}",
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
            
        rl_copp_obj.terminal_pid = f"CoppeliaTerminal_{timestamp}"

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
        client = RemoteAPIClient(port=zmq_port)
        rl_copp_obj.current_sim = client.getObject('sim')
    except Exception as e:
        logging.error(f"It was not possible to connect with CoppeliaSim: {e}")
        if process:
            process.terminate()
        return False

    logging.info("Connection established with CoppeliaSim")

    # Check if scene is loaded
    if is_scene_loaded(rl_copp_obj.current_sim, rl_copp_obj.args.scene_path):
        logging.info("Scene is already loaded, simulation will be stopped in case that it's running...")
        stop_coppelia_simulation(rl_copp_obj.current_sim)
    else:
        logging.info(f"Loading scene: {rl_copp_obj.args.scene_path}")
        rl_copp_obj.current_sim.loadScene(rl_copp_obj.args.scene_path)
        logging.info("Scene loaded successfully.")

    # Create target's discs # TODO
    create_discs_under_target(rl_copp_obj)

    # Update code inside Coppelia's scene
    update_and_copy_script(rl_copp_obj)

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
        return [], {}

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
