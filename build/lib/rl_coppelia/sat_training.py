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

    It systematically varies the `fixed_actime` parameter in the configuration file to evaluate
    its effect on the training performance, helping determine the optimal value.

Usage:
    rl_coppelia sat_training --robot_name <robot_name>
                              --session_name <session_name>
                              --base_params_file <path_to_base_params>
                              [--dis_parallel_mode]
                              [--max_workers <num_workers>]
                              [--start_value <float>]
                              [--end_value <float>]
                              [--increment <float>]
                              [--verbose <0|1|2|3>]

Features:
    - Automatically creates session-specific parameter files varying the fixed action time.
    - Executes training runs in parallel or sequentially based on user preference.
    - Limits concurrent processes using semaphores to avoid CoppeliaSim conflicts.
    - Introduces automatic delays between parallel job submissions to reduce overlap.
    - Aggregates and saves training results (status and duration) into a timestamped summary CSV.
    - Logs errors, warnings, and exceptions encountered during any training run.
"""

import csv
import datetime
import logging
import os
import time

import concurrent.futures
from common import utils
from common.rl_coppelia_manager import RLCoppeliaManager
from concurrent.futures import ThreadPoolExecutor
from threading import Semaphore



def limited_auto_run(rl_manager, file, semaphore):
    with semaphore:
        return utils.auto_run_mode(rl_manager.args, "sampling_at", file, no_gui=True)
    

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

    # Manage semaphore for avoiding the experiments to collide between them
    max_workers = rl_copp.args.max_workers
    semaphore = Semaphore(max_workers)
    
    # If parallel mode is enabled, run the trainings in parallel
    if not rl_copp.args.dis_parallel_mode:
        logging.info(f"Running {len(param_files)} trainings in parallel with max_workers={max_workers}")

        futures = []
        with ThreadPoolExecutor() as executor:
            for file in param_files:
                future = executor.submit(limited_auto_run, rl_copp, file, semaphore)
                futures.append((future, file))
                logging.info(f"Submitted job for {os.path.basename(file)}, waiting 40 seconds before next submission...")
                time.sleep(40)

            for future, file in futures:
                try:
                    results.append(future.result())
                except Exception as exc:
                    file_name = os.path.basename(file)
                    logging.error(f"{file_name} generated an exception: {exc}")
                    fixed_actime = os.path.basename(file).split('_')[-1].replace('.json', '')
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