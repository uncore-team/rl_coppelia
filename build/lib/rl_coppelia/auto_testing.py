"""
Project: Robot Training and Testing RL Algorithms in CoppeliaSim
Author: Adrián Bañuls Arias
Version: 1.0
Date: 2025-03-25
License: GNU General Public License v3.0

Description:
    This script manages the automatic evaluation of multiple trained reinforcement learning models 
    within a CoppeliaSim environment. It executes testing sessions for each model listed, using the 
    specified number of iterations per model, and stores comparative results in a summary CSV file.

    The script supports sequential or parallel execution to adapt to system capabilities, ensuring 
    efficient and scalable performance testing for different model configurations.

Usage:
    rl_coppelia auto_testing --robot_name <robot_name> 
                             --session_name <session_name> 
                             --model_ids <id1> <id2> ... 
                             --iterations <num_iterations> 
                             [--dis_parallel_mode] 
                             [--max_workers <num_workers>] 
                             [--verbose <0|1|2|3>]

Features:
    - Automatically locates and loads models by their ID.
    - Executes a fixed number of test episodes per model.
    - Supports both sequential and parallel testing of models.
    - Limits concurrent testing jobs via configurable worker count.
    - Captures performance metrics and saves results in a timestamped CSV summary.
    - Provides configurable verbosity and debug logging.
"""

from argparse import Namespace
import csv
import datetime
import glob
import logging
import os

from rl_coppelia import plot
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys
import time
import concurrent.futures
from common import utils
from common.rl_coppelia_manager import RLCoppeliaManager


def copy_latest_test_csv(rl_copp_obj, model_name, session_folder):
    """
    Copies the latest test and otherdata CSV files of a given model into the current auto-test session folder.

    This function looks for the latest CSV files generated during testing for the specified model
    (both 'test' and 'otherdata' files), and copies them to the session folder with a renamed filename
    for consolidation.

    Args:
        model_name (str): Full name of the model (e.g., 'turtleBot_model_319').
        session_folder (str): Path to the folder where the test results should be copied.

    Returns:
        tuple[str or None, str or None]: Paths to the copied test and otherdata CSV files, or None if not found.
    """
    testing_path = os.path.join(rl_copp_obj.base_path, "robots", rl_copp_obj.args.robot_name, "testing_metrics")
    logging.info(f"testing_path: {testing_path}")

    test_files = sorted(
        glob.glob(
            os.path.join(testing_path, f"{model_name}_testing", f"{model_name}_test_*.csv")
        )
    )

    otherdata_files = sorted(
        glob.glob(
            os.path.join(testing_path, f"{model_name}_testing", f"{model_name}_otherdata_*.csv")
        )
    )

    test_csv_path = None
    otherdata_csv_path = None

    if test_files:
        latest_test_csv = test_files[-1]
        test_csv_path = os.path.join(session_folder, f"{model_name}_test.csv")
        os.system(f"cp '{latest_test_csv}' '{test_csv_path}'")
        logging.info(f"Copied test CSV to session folder: {test_csv_path}")

    if otherdata_files:
        latest_otherdata_csv = otherdata_files[-1]
        otherdata_csv_path = os.path.join(session_folder, f"{model_name}_otherdata.csv")
        os.system(f"cp '{latest_otherdata_csv}' '{otherdata_csv_path}'")
        logging.info(f"Copied otherdata CSV to session folder: {otherdata_csv_path}")

    if test_csv_path or otherdata_csv_path:
        return test_csv_path, otherdata_csv_path

    logging.warning(f"No test or otherdata CSV found for model: {model_name}")
    return None, None


def collect_test_records(rl_copp_obj, session_folder):
    """
    Collects the latest test records for each model and merges them with training data to include action time.

    This function extracts the latest matching row for each model from 'test_records.csv' (testing metrics),
    and obtains the 'Action time (s)' from 'train_records.csv' (training metrics). It builds a combined 
    session-specific CSV with selected fields and sorts it by action time.

    Args:
        rl_copp_obj (RLCoppeliaManager): Manager object containing robot name and model IDs.
        session_folder (str): Directory where the filtered CSV will be saved.
    """
    base_robot_path = os.path.join(rl_copp_obj.base_path, "robots", rl_copp_obj.args.robot_name)
    testing_records_file = os.path.join(base_robot_path, "testing_metrics", "test_records.csv")
    training_records_file = os.path.join(base_robot_path, "training_metrics", "train_records.csv")
    session_records = os.path.join(session_folder, f"test_{os.path.basename(session_folder)}.csv")

    # Columns to extract from test_records.csv (except Action time, which comes from training)
    selected_columns = [
        "Exp_id", "Action time (s)", "Avg reward", "Avg time reach target",
        "Number of collisions", "Target zone 1 (%)", "Target zone 2 (%)",
        "Target zone 3 (%)", "Average distance per episode (m)"
    ]

    rows = []
    for model_id in rl_copp_obj.args.model_ids:
        model_name = f"{rl_copp_obj.args.robot_name}_model_{model_id}"

        # Get action time from training records
        action_time = utils.get_data_from_training_csv(
            model_name, training_records_file, column_header="Action time (s)"
        )

        # Get latest matching row from testing records
        with open(testing_records_file, newline='') as f:
            reader = csv.DictReader(f)
            matching_rows = [row for row in reader if row.get("Exp_id", "").startswith(model_name)]
            if matching_rows:
                last_row = matching_rows[-1]
                filtered_row = {key: last_row.get(key, "") for key in selected_columns if key != "Action time (s)"}
                filtered_row["Action time (s)"] = action_time
                rows.append(filtered_row)

    # Helper for safe float sorting
    def safe_float(val):
        try:
            return float(val)
        except:
            return float('inf')

    # Sort rows by action time
    rows.sort(key=lambda r: safe_float(r.get("Action time (s)", "inf")))

    # Save filtered and sorted results
    if rows:
        with open(session_records, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=selected_columns)
            writer.writeheader()
            writer.writerows(rows)
        logging.info(f"Saved session test summary to {session_records}")



def generate_plots(rl_copp_obj, session_folder):
    """
    Generates plots for the given models and saves them to the session folder.

    This function configures the plotting arguments and temporarily changes the working directory 
    to the session folder so that all output figures are saved directly within it. It then 
    invokes the central plot manager with the appropriate arguments.

    Args:
        robot_name (str): Name of the robot.
        model_ids (list[int]): List of model IDs to include in the plots.
        session_folder (str): Directory where all plots will be saved.
        verbose (int): Verbosity level for logging and debugging.
    """
    plots_dir = os.path.join(session_folder, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_args = Namespace(
        robot_name=rl_copp_obj.args.robot_name,
        model_ids=rl_copp_obj.args.model_ids,
        scene_to_load_folder=None,
        plot_types=["convergence-all", "compare-rewards", "compare-convergences", "grouped_bar_speeds", "plot_boxplots"],
        verbose=rl_copp_obj.args.verbose,
        save_plots = True
    )

    # Switch working directory to session folder to ensure plots are saved there
    original_dir = os.getcwd()
    os.chdir(plots_dir)
    try:
        plot.main(plot_args)
    finally:
        # Restore original working directory after plotting
        os.chdir(original_dir)



def main(args):
    """
    Executes multiple testing runs. This method allows the user to test multiple models just by indicating
    a list of model numbers.
    """

    rl_copp = RLCoppeliaManager(args)

    # Base path for the auto testing results
    auto_test_base_path = os.path.join(rl_copp.base_path, "robots", rl_copp.args.robot_name, "auto_test")

    if args.session_name is None:   # Create the directory for the new auto-test session.
        session_dir, session_id = utils.create_next_auto_test_folder(auto_test_base_path)

    else:
        # Remove '.csv' if present
        session_name_clean = args.session_name[:-4] if args.session_name.endswith('.csv') else args.session_name
        session_dir = os.path.join(auto_test_base_path, session_name_clean)

        os.makedirs(session_dir, exist_ok=True)

        # Extract session_id from cleaned session name
        session_filename = os.path.basename(session_name_clean)
        parts = session_filename.split('_')
        if len(parts) > 1:
            session_id = parts[-1]
        else:
            session_id = session_filename

    logging.info(f"Session directory: {session_dir}")
    logging.info(f"Session ID: {session_id}")
    
    results = []
    
    # If parallel mode is enabled, run the trainings in parallel
    if not rl_copp.args.dis_parallel_mode:
        logging.info(f"Running {len(args.model_ids)} testings in parallel with max_workers={rl_copp.args.max_workers}")

        # Submit all training jobs with a delay between submissions.
        futures = []            
        with concurrent.futures.ProcessPoolExecutor(max_workers=rl_copp.args.max_workers) as executor:
            for model_id in args.model_ids:
                futures.append(executor.submit(utils.auto_run_mode, rl_copp.args, "auto_testing", model_id=model_id, no_gui=True))
                logging.info(f"Submitted job for model {model_id}, waiting 8 seconds before next submission...")
                time.sleep(12)
            
            # Collect results as they complete.
            for future in concurrent.futures.as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as exc:
                    model_id = "unknown"
                    for submitted_future, mid in zip(futures, args.model_ids):
                        if submitted_future == future:
                            model_id = mid
                            break
                    logging.info(f"Model {model_id} generated an exception: {exc}")
                    results.append((model_id, "Exception", 0))

    # If parallel mode is not enabled, run the trainings sequentially
    else:
        logging.info(f"Running {len(args.model_ids)} testings sequentially")
        for model_id in args.model_ids:
            logging.info("Testing a new model in a few seconds...")
            result = utils.auto_run_mode(rl_copp.args, "auto_testing", model_id=model_id, no_gui=True)
            results.append(result)
            time.sleep(2)

    # Copy individual test CSVs
    for m_name, status, _ in results:
        if status == "Success":
            copy_latest_test_csv(rl_copp, m_name, session_dir)

    # Generate combined session CSV and plots
    collect_test_records(rl_copp, session_dir)
    generate_plots(rl_copp, session_dir)

    logging.info("All tests completed, records consolidated, and plots generated.")


if __name__ == "__main__":
    main()