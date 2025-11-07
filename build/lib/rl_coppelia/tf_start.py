"""
Project: Robot Training and Testing RL Algorithms in CoppeliaSim
Author: Adrián Bañuls Arias
Version: 1.0
Date: 2025-03-25
License: GNU General Public License v3.0

Description:
    This script launches TensorBoard to visualize the training metrics of a specified reinforcement learning model.
    It parses the model name, locates the corresponding TensorBoard logs directory, and opens the TensorBoard interface
    in the default web browser once the data becomes available.

Usage:
    rl_coppelia tf_start --model_name <model_name>
                         [--verbose <0|1|2|3>]
Features:
    - Automatically launches a CoppeliaSim instance for testing.
    - Loads a preconfigured scene and initializes the corresponding environment.
    - Supports testing multiple models across multiple targets in a single execution.
    - Dynamically assigns models to scene targets based on model count.
    - Automatically detects and loads trained models using the correct SB3 algorithm.
    - Measures and logs key performance metrics such as reward, time, crashes, and target zone.
    - Saves detailed episode metrics and a summary CSV with aggregated statistics.
    - Calculates episode trajectory distance if no collision is detected.
    - Cleans up and stops CoppeliaSim after testing is complete.
"""

import logging
import os
import re
import subprocess
import sys
import threading
import time
import webbrowser
import requests
from common.rl_coppelia_manager import RLCoppeliaManager


def wait_for_scalars_ready(base_url, timeout=30):
    print("Waiting for scalars to become available...", end="", flush=True)
    endpoint = f"{base_url}/data/plugin/scalars/tags"
    for _ in range(timeout * 2):
        try:
            r = requests.get(endpoint)
            if r.status_code == 200 and r.json():  # must return non-empty JSON
                print(" ready.")
                return True
        except Exception:
            pass
        print(".", end="", flush=True)
        time.sleep(0.5)
    print("\n[ERROR] Scalars not found in expected time.")
    return False


def run_tensorboard(curr_tf_logs_path):
    command = [
        "tensorboard",
        "--logdir", curr_tf_logs_path,
        "--port=0",
        "--load_fast=false"
    ]

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    url = None
    while True:
        line = process.stdout.readline()
        if not line:
            break
        print(line.strip())

        if "http://localhost" in line and url is None:
            match = re.search(r'(http://localhost:\d+)', line)
            if match:
                url = match.group(1)

                # Wait until scalars are visible
                if wait_for_scalars_ready(url):
                    webbrowser.open(url)
                break

    print("Press 'q' and Enter to stop TensorBoard.")
    if input().lower().strip() == "q":
        process.terminate()
        process.wait()


def main(args):
    """
    Automates the process of launching TensorBoard for a given model and opening it in the browser.

    This function takes a model name of the form <robot_name>_model_<id>, extracts the robot name and 
    model ID, searches for the corresponding TensorBoard logs folder, and opens TensorBoard in the browser.

    Args:
        model_name (str): The model name in the format <robot_name>_model_<id> (e.g. "turtleBot_model_15").
    
    Raises:
        Exception: If the TensorBoard logs folder does not exist or there is an issue running TensorBoard.
    """
    rl_copp = RLCoppeliaManager(args)

    # Extract robot name and model id from the model_name
    try:
        # Extrae solo el nombre base (sin subcarpetas ni sufijos)
        base_name = os.path.basename(rl_copp.args.model_name)
        # Usa regex para extraer robot_name y model_id
        match = re.match(r"([a-zA-Z0-9]+)_model_(\d+)", base_name)
        if match:
            robot_name = match.group(1)
            model_id = match.group(2)
        else:
            raise ValueError
    except Exception:
        logging.error(f"Error: Invalid model name format. Expected format like '<robot_name>_model_<id>' or '<robot_name>_model_<id>_last', but got '{rl_copp.args.model_name}'")
        return
    
    # Define path
    tf_logs_path = rl_copp.paths["tf_logs"]
    curr_tf_logs_path = os.path.join(tf_logs_path, f'{robot_name}_tflogs_{model_id}')

    logging.info(f"Checking in this directory: {curr_tf_logs_path}")
    
    # Check if the folder exists
    if os.path.isdir(curr_tf_logs_path):
        logging.info(f"Found TensorBoard logs at: {curr_tf_logs_path}")
        
        # Run TensorBoard in a separate thread so that the program doesn't block
        tensorboard_thread = threading.Thread(target=run_tensorboard, args=(curr_tf_logs_path,))
        tensorboard_thread.start()
    else:
        logging.critical(f"Warning: Something failed while checking in this directory: {curr_tf_logs_path}. TensorBoard logs not found for model {model_id} of robot {robot_name}. Check your spelling!")
        sys.exit()


if __name__ == "__main__":
    main()