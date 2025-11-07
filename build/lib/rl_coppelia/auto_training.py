"""
Project: Robot Training and Testing RL Algorithms in CoppeliaSim
Author: Adrián Bañuls Arias
Version: 1.0
Date: 2025-03-25
License: GNU General Public License v3.0

Description:
    This script manages the execution of a special training mode for robots in a CoppeliaSim 
    environment. It automates the discovery of configuration files (JSON), and executes 
    multiple training sessions accordingly. The method is useful for testing different 
    training configurations efficiently and reproducibly.

    Executes multiple training runs using pre-defined parameter files located inside a session
    directory. Supports both sequential and parallel execution, and stores the results in a
    summary CSV file.

Usage:
    rl_coppelia auto_training --robot_name <robot_name> --session_name <session_name>
                               [--dis_parallel_mode] 
                               [--max_workers <num>] 
                               [--timestamp <timestamp>] 
                               [--verbose <0|1|2|3>]

Features:
    - Automatically detects and loads all param.json files from the session directory.
    - Supports parallel or sequential training execution based on user preference.
    - Applies configurable delay between process submissions to avoid collisions.
    - Tracks and stores the status and duration of each training in a CSV file.
    - Ensures reproducibility by preserving training metadata and timestamps.
    - Helps benchmark different training settings easily in batch mode.
"""

import csv
import datetime
import glob
import logging
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys
import time
import concurrent.futures
from common import utils
from common.rl_coppelia_manager import RLCoppeliaManager


def main(args):
    """
    Executes multiple training runs using pre-defined parameter files. This method allows the user to 
    test multiple settings just by preparing some json files. The code will automatically execute a 
    training per param.json file.

    The function discovers all parameter files in the specified session directory, and manages the training process
    either sequentially or in parallel. It also generates a CSV summary file of the training results.
    """

    rl_copp = RLCoppeliaManager(args)

    # Get the directory containing the parameter files for the session.
    session_dir = os.path.join(rl_copp.base_path, "robots", rl_copp.args.robot_name, "auto_trainings", rl_copp.args.session_name)
        
    # Create the directory if it doesn't exist
    os.makedirs(session_dir, exist_ok=True)

    # Check if the directory is empty
    if not os.listdir(session_dir):
        logging.critical(f"ERROR: The directory {session_dir} is empty. Please add the desired param.json files for training.")
        sys.exit()
    
    # Search all the json files inside the provided folder.
    param_files = glob.glob(os.path.join(session_dir, "*.json"))

    # Path to the CSV File with the summary of the chained training.
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_csv = os.path.join(session_dir, f"training_summary_{rl_copp.args.robot_name}_{rl_copp.args.session_name}_{timestamp}.csv")
    
    results = []
    
    # If parallel mode is enabled, run the trainings in parallel
    if not rl_copp.args.dis_parallel_mode:
        logging.info(f"Running {len(param_files)} trainings in parallel with max_workers={rl_copp.args.max_workers}")

        # Submit all training jobs with a delay between submissions.
        futures = []            
        with concurrent.futures.ProcessPoolExecutor(max_workers=rl_copp.args.max_workers) as executor:
            for file in param_files:
                futures.append(executor.submit(utils.auto_run_mode, rl_copp.args, "auto_training", file, no_gui=True))
                logging.info(f"Submitted job for {os.path.basename(file)}, waiting {8} seconds before next submission...")
                time.sleep(8)  # Wait before starting the next process.
            
            # Collect results as they complete.
            for future in concurrent.futures.as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as exc:
                    file_name = "unknown"
                    for f, submitted_future in zip(param_files, futures):
                        if submitted_future == future:
                            file_name = os.path.basename(f)
                            break
                    logging.info(f"{file_name} generated an exception: {exc}")

                    results.append((file_name, "Exception", 0))

    # If parallel mode is not enabled, run the trainings sequentially
    else:
        logging.info(f"Running {len(param_files)} trainings sequentially")
        for file in param_files:
            logging.info("Training a new model in a few seconds...")
            result = utils.auto_run_mode(rl_copp.args, "auto_training", file=file, no_gui=True)
            results.append(result)
            time.sleep(2)

    # Sort the results using the first column (params file name).
    results_sorted = sorted(results, key=lambda x: x[0])
    
    # Write results to CSV
    with open(summary_csv, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Param File", "Status", "Duration (hours)"])
        writer.writerows(results_sorted)
    
    logging.info(f"Training summary saved to {summary_csv}")


if __name__ == "__main__":
    main()