"""
Project: Robot Training and Testing RL Algorithms in CoppeliaSim
Author: Adrián Bañuls Arias
Version: 1.0
Date: 2025-03-25
License: GNU General Public License v3.0

Description:
    This script manages the execution of a set of tests for a robot in a CoppeliaSim 
    environment. It automates the testing process, running a test for each specified model, and
    comparing and saving the results.

Usage:
    rl_coppelia auto_test --robot_name <robot_name> --model_ids <model_ids> 
                            --iterations <num> [--parallel_mode] [--max_workers]
                            [--verbose <num>]


Features:
    - Automatically creates required directories if they do not exist.
    - Runs a testing session either sequentially or in parallel with a delay between submissions.
    - Saves a summary of testing results in a CSV file, along with some plots comparing the obtained metrics.
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
    Executes multiple testing runs. This method allows the user to test multiple models just by indicating
    a list of model numbers.
    """

    rl_copp = RLCoppeliaManager(args)

    # Create the directory for the new auto-test session.
    session_dir, session_id = utils.create_next_auto_test_folder(rl_copp.base_path, "robots", rl_copp.args.robot_name, "auto_test")

    # Path to the CSV File with the summary of the auto testing session.
    summary_csv = os.path.join(session_dir, f"testing_summary_{rl_copp.args.robot_name}_{session_id}.csv")
    
    results = []
    
    # If parallel mode is enabled, run the trainings in parallel
    if not rl_copp.args.dis_parallel_mode:
        logging.info(f"Running {len(args.model_ids)} testings in parallel with max_workers={rl_copp.args.max_workers}")

        # Submit all training jobs with a delay between submissions.
        futures = []            
        with concurrent.futures.ProcessPoolExecutor(max_workers=rl_copp.args.max_workers) as executor:
            for id in args.model_ids:
                futures.append(executor.submit(utils.auto_run_mode, rl_copp.args, "auto_testing", model_id = args.model_ids[id], no_gui=True))
                logging.info(f"Submitted job for model {args.model_ids[id]}, waiting {8} seconds before next submission...")
                time.sleep(8)  # Wait before starting the next process.
            
            # Collect results as they complete.
            for future in concurrent.futures.as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as exc:
                    model_id = "unknown"
                    for submitted_future, current_model in zip(futures, args.model_ids):
                        if submitted_future == future:
                            model_id = args.model_ids[current_model]
                            break
                    logging.info(f"Model {model_id} generated an exception: {exc}")
                    results.append((model_id, "Exception", 0))

    # If parallel mode is not enabled, run the trainings sequentially
    else:
        logging.info(f"Running {len(args.model_ids)} testings sequentially")
        for id in args.model_ids:
            logging.info("Testing a new model in a few seconds...")
            result = utils.auto_run_mode(rl_copp.args, "auto_testing", model_id = args.model_ids[id], no_gui=True)
            results.append(result)
            time.sleep(2)

    # Sort the results using the first column (params file name).
    results_sorted = sorted(results, key=lambda x: x[0])
    
    # Write results to CSV
    with open(summary_csv, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Model name", "Status", "Duration (hours)"])
        writer.writerows(results_sorted)
    
    logging.info(f"Testing summary saved to {summary_csv}")


if __name__ == "__main__":
    main()