"""
Project: Robot Training and Testing RL Algorithms in CoppeliaSim
Author: Adrián Bañuls Arias
Version: 1.0
Date: 2025-03-25
License: GNU General Public License v3.0

Usage:
    rl_coppelia save --model_name <robot_name>_model_<id>
                     --new_name <new_model_name>
                     [--verbose <0|1|2|3>]

Features:
    - Validates the format of the model name to extract robot name and ID.
    - Gathers model weights, training callbacks, training/test metrics, TensorBoard logs, and parameter configuration.
    - Automatically renames internal files to match the new name.
    - Filters test results and training logs relevant to the specified model.
    - Saves the complete dataset into a single ZIP file under the `results/` folder.
    - Keeps a log file mapping original names to the new names for traceability.
    - Useful for model sharing, versioning, or final packaging before deployment.
"""

import datetime
import logging
import os
import re
import sys
import zipfile

import pandas as pd
from common.rl_coppelia_manager import RLCoppeliaManager


def main(args):
    """
    Saves the model, callbacks, training metrics, test results, and parameters used in a zip file.
    The zip file will be stored in a 'results' folder within the base path.

    It just needs the following inputs:
        - model_name (str): The name of the model (should be '<robot_name>_model_ID').
        - new_name (str): New name to save the model.
    """
    rl_copp = RLCoppeliaManager(args)
    
    # Extract model_id from model_name (assuming model_name is in the format '<robot_name>_model_<ID>')
    parts = rl_copp.args.model_name.rsplit("_")  # Divide from the last "_"
    if len(parts) != 3 or not parts[2].isdigit():
        logging.critical(f"ERROR: The model name provided was {rl_copp.args.model_name}, and it needs to follow this pattern <robot_name>_model_<ID>")
        sys.exit()

    robot_name, model_id = parts[0], parts[2]
    
    # Get for the model and related files
    models_path = rl_copp.paths["models"]
    callbacks_path = rl_copp.paths["callbacks"]
    train_log_path = rl_copp.paths["tf_logs"]
    training_metrics_path = rl_copp.paths["training_metrics"]
    testing_metrics_path = rl_copp.paths["testing_metrics"]
    parameters_used_path = rl_copp.paths["parameters_used"]

    # Ensure the 'results' directory exists
    results_dir = os.path.join(rl_copp.base_path, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Define the zip file name
    zip_filename = f'results_{rl_copp.args.new_name}.zip'
    zip_filepath = os.path.join(results_dir, zip_filename)
    
    # Create the zip file
    with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add the model .zip file
        model_zip_path = os.path.join(models_path, f'{robot_name}_model_{model_id}.zip')
        logging.info(f"Searching model inside {model_zip_path}")
        if os.path.exists(model_zip_path):
            zipf.write(model_zip_path, arcname=f'{rl_copp.args.new_name}.zip')
        else:
            logging.warning(f"Warning: Model file {rl_copp.args.model_name}.zip not found.")
        
        # Add the callback folder
        callback_folder_path = os.path.join(callbacks_path, f'{robot_name}_callbacks_{model_id}')   # To search
        callback_folder_path_new = os.path.join(callbacks_path, f'{robot_name}_callbacks_{rl_copp.args.new_name}')  # To save
        logging.info(f"Searching callbacks inside {callback_folder_path}")
        if os.path.isdir(callback_folder_path):
            for root, _, files in os.walk(callback_folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.join(os.path.basename(callback_folder_path_new), os.path.relpath(file_path, start=callback_folder_path))
                    zipf.write(file_path, arcname=arcname)
        else:
            logging.warning(f"Warning: Callback folder for model ID {model_id} not found.")
        
        # Add the parameters file
        params_file_path = None
        for param_file in os.listdir(parameters_used_path):
            if re.search(r'model_(\d+)', param_file) and f'model_{model_id}' in param_file and param_file.endswith('.json'):
                params_file_path = os.path.join(parameters_used_path, param_file)
                break   # We assume that there will be just a json file per model.

        if params_file_path:
            logging.info(f"Searching parameters files inside {params_file_path}")
            zipf.write(params_file_path, arcname=f'params_file_{rl_copp.args.new_name}.json')
        else:
            logging.warning(f"Warning: Parameters file for model ID {model_id} not found.")
        
        # Add all test result CSV files
        logging.info(f"Searching test files inside {testing_metrics_path}")
        test_files = [
            f for f in os.listdir(testing_metrics_path)
            if f'{robot_name}_model_{model_id}_test' in f and f.endswith('.csv')
        ]

        if test_files:
            for test_file in test_files:
                test_file_path = os.path.join(testing_metrics_path, test_file)
                test_file_new = re.sub(r'model_\d+', f'{rl_copp.args.new_name}', test_file)
                zipf.write(test_file_path, arcname=os.path.join("tests", test_file_new)) 
        else:
            logging.warning(f"Warning: No test metrics files found for model ID {model_id}.")

        # Process test_records.csv
        test_records_path = os.path.join(testing_metrics_path, "test_records.csv")

        if os.path.exists(test_records_path):
            logging.info(f"Filtering test_records.csv for model ID {model_id}")

            # Load CSV into a DataFrame
            df = pd.read_csv(test_records_path)

            # Filtrate the rows that corresponds to the current model
            filtered_df = df[df.iloc[:, 0].isin(test_files)]

            if not filtered_df.empty:
                # Replace the old model name with the new one in all the rows (once they have been filtered)
                filtered_df.iloc[:, 0] = filtered_df.iloc[:, 0].apply(
                    lambda name: re.sub(r'model_\d+', f'{rl_copp.args.new_name}', name)
                )
                
                # Save the filtrated CSV temporarily
                filtered_csv_path = os.path.join(testing_metrics_path, "filtered_test_records.csv")
                filtered_df.to_csv(filtered_csv_path, index=False)

                zipf.write(filtered_csv_path, arcname=os.path.join("tests", f"test_records_{rl_copp.args.new_name}.csv"))

                # Remove temporal file
                os.remove(filtered_csv_path)
            else:
                logging.warning(f"Warning: No matching entries found in test_records.csv for model ID {model_id}.")
        else:
            logging.warning("Warning: test_records.csv not found in testing_metrics directory.")
        
        # Add the training metrics CSV file
        logging.info(f"Searching training metrics files inside {training_metrics_path}")
        found_files = False

        for train_file in os.listdir(training_metrics_path):
            if f'{robot_name}_model_{model_id}_train' in train_file and train_file.endswith('.csv'):
                train_file_path = os.path.join(training_metrics_path, train_file)
                train_file_new = re.sub(r'model_\d+', f'{rl_copp.args.new_name}', train_file)
                zipf.write(train_file_path, arcname=train_file_new)
                found_files = True
        if not found_files:
            logging.warning(f"Warning: No train metrics files found for model ID {model_id}.")

        # Add the training logs Tensorboard folder
        tflogs_path = os.path.join(train_log_path, f'{robot_name}_tflogs_{model_id}')
        tflogs_path_new = os.path.join(train_log_path, f'{robot_name}_tflogs_{rl_copp.args.new_name}')
        print(tflogs_path_new)
        logging.info(f"Searching tflogs inside {tflogs_path}")
        if os.path.isdir(tflogs_path):
            for root, dirs, files in os.walk(tflogs_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, tflogs_path)
                    print(arcname)
                    zipf.write(file_path, arcname=f'tflogs/{arcname}')
        else:
            logging.warning(f"Warning: TensorBoard logs for model ID {model_id} not found.")
                
    logging.info(f"All files have been saved to {zip_filepath}")

    # Save the old name just in case you need it in the future
    original_name_txt_path = os.path.join(results_dir, "original_model_name.txt")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(original_name_txt_path, 'a') as f:
        f.write(f"[{timestamp}] {robot_name}_model_{model_id} --> {robot_name}_{rl_copp.args.new_name}\n")


if __name__ == "__main__":
    main()