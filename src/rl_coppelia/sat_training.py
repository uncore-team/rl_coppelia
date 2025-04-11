"""
Project: Robot Training and Testing RL Algorithms in CoppeliaSim
Author: Adrián Bañuls Arias
Version: 1.0
Date: 2025-03-25
License: GNU General Public License v3.0

Description:
    This script manages the execution of a special training mode for robots in a CoppeliaSim 
    environment. It automates the generation of parameter files, the execution of training 
    runs (either sequentially or in parallel), and logs the results into a summary CSV file.

    Generates parameter files by varying the "fixed_actime" value, executes training runs, and 
    logs the results. This method is used when you want to find the optimal action time for training a robot using RL algorithms.

Usage:
    rl_coppelia sat_training --robot_name <robot_name> --session_name <session_name> 
                               [--parallel_mode] [--max_workers <num>] [--base_params_file <path>]
                               [--start_value <float>] [--end_value <float>] [--increment <float>]

Features:
    - Automatically creates required directories if they do not exist.
    - Cleans the output directory of old JSON files before generating new ones.
    - Runs training sessions either sequentially or in parallel with a delay between submissions.
    - Detects and terminates CoppeliaSim instances after training.
    - Saves a summary of training results in a CSV file.
"""

import csv
import datetime
import logging
import os
import time

import concurrent.futures
from common import utils
from common.rl_coppelia_manager import RLCoppeliaManager


def main(args):
    """
    Performs a several trainings for sampling the optimal action time for an agent using a custom environment.
    Generates parameter files by varying the "fixed_actime" value, executes training runs, and logs the results. 
    This method is used when you want to find the optimal action time for training a robot using RL algorithms.
    """

    rl_copp = RLCoppeliaManager(args)

    # Get the directory containing the parameter files for the session.
    session_dir = os.path.join(rl_copp.base_path, "robots", rl_copp.args.robot_name, "sat_trainings", rl_copp.args.session_name)
    
    # Create the directory if it doesn't exist
    os.makedirs(session_dir, exist_ok=True)

    # Create parameter files
    logging.info("Create param files")
    param_files = utils.auto_create_param_files(
        os.path.join(rl_copp.base_path, "configs", rl_copp.args.base_params_file),
        session_dir,
        rl_copp.args.start_value,
        rl_copp.args.end_value,
        rl_copp.args.increment
    )   
        
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
                futures.append(executor.submit(utils.auto_run_mode, rl_copp.args, "sampling_at", file, no_gui=True))
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
                    logging.error(f"{file_name} generated an exception: {exc}")

                    fixed_actime = os.path.basename(file_name).split('_')[-1].replace('.json', '')
                    results.append((file_name, fixed_actime, "Exception", 0))

    # If parallel mode is not enabled, run the trainings sequentially
    else:
        logging.info(f"Running {len(param_files)} trainings sequentially")
        for file in param_files:
            logging.info("Training a new model in a few seconds...")
            result = utils.auto_run_mode (rl_copp.args, "sampling_at", file=file, no_gui=True)
            results.append(result)
            time.sleep(2)
    
    # Sort the results using the first column (params file name).
    results_sorted = sorted(results, key=lambda x: x[0])

    # Write results to CSV
    with open(summary_csv, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Param File", "Action time (s)", "Status", "Duration (hours)"])
        writer.writerows(results_sorted)
    
    logging.info(f"Training summary saved to {summary_csv}")


if __name__ == "__main__":
    main()