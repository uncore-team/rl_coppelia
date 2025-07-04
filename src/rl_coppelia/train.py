import csv
import datetime
import glob
import inspect
import logging
import os
import shutil

import numpy as np
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # If you don't want logs from Tensorflow to be displayed
import sys
import time
import stable_baselines3
from common import utils
from common.rl_coppelia_manager import RLCoppeliaManager
from stable_baselines3.common.callbacks import CheckpointCallback
from tensorboard.backend.event_processing import event_accumulator
import threading
import select
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.evaluation import evaluate_policy
import traceback


def main(args):
    """
    Train the model using a custom environment.

    This function creates an environment, starts a CoppeliaSim instance and trains an agent 
    using that environment. Finally, it closes the opened simulation.
    """
    rl_copp = RLCoppeliaManager(args)

    ### Start CoppeliaSim instance
    rl_copp.start_coppelia_sim()

    ### Create the environment
    rl_copp.create_env()

    ### Train the model
    # Extract the needed paths for training
    models_path = rl_copp.paths["models"]
    callbacks_path = rl_copp.paths["callbacks"]
    train_log_path = rl_copp.paths["tf_logs"]
    training_metrics_path = rl_copp.paths["training_metrics"]
    parameters_used_path = rl_copp.paths["parameters_used"]
    train_log_file_path = os.path.join(train_log_path,f"{rl_copp.args.robot_name}_tflogs_{rl_copp.file_id}")

    logging.info(f"Training mode. Final trained model will be saved in {models_path}")
    logging.info(f"EXPERIMENT ID: {rl_copp.file_id}")

    # Get final model name
    to_save_model_path, model_name = utils.get_next_model_name(models_path, rl_copp.args.robot_name, rl_copp.file_id)

    # Callback function to save the model every x timesteps
    to_save_callbacks_path, _ = utils.get_next_model_name(callbacks_path, rl_copp.args.robot_name, rl_copp.file_id, callback_mode=True)
    checkpoint_callback = CheckpointCallback(save_freq=rl_copp.params_train['callback_frequency'], save_path=to_save_callbacks_path, name_prefix=rl_copp.args.robot_name)

    # Callback function for stopping the learning process if a specific key is pressed
    stop_callback = utils.StopTrainingOnKeypress(key="F") 

    # Callback for saving the best model based on the training reward every x timesteps
    eval_train_callback = utils.SaveOnBestTrainingRewardCallback(
        check_freq=1000,    # default: 1000
        save_path=to_save_model_path, 
        log_dir=rl_copp.log_monitor, 
        verbose=1)

    # Callback for evaluating and saving the best model based on the testing reward every x timesteps
    # eval_test_callback = utils.CustomEvalCallback(
    #     rl_copp.env,
    #     best_model_save_path=to_save_model_path,
    #     eval_freq=250, # default: 1500
    #     deterministic=True,
    #     render=False,
    #     rl_manager=rl_copp 
    # )

    # Callback for logging custom metrics in Tensorboard
    metrics_callback = utils.CustomMetricsCallback(rl_copp, total_timesteps=rl_copp.params_train["total_timesteps"])

    # Get the training algorithm from the parameters file
    try:
        ModelClass = getattr(stable_baselines3, rl_copp.params_train["sb3_algorithm"])
    except:
        logging.error(f"Algorithm indicated in parameters file (json) is not valid. Parameter read: {rl_copp.params_train['sb3_algorithm']} ")
        raise

    # Configure the model
    init_params = inspect.signature(ModelClass.__init__).parameters
    if "n_steps" in init_params:
        model = ModelClass(
            policy = rl_copp.params_train["policy"], 
            env = rl_copp.env,
            n_steps = rl_copp.params_train["n_training_steps"], 
            verbose=True, 
            tensorboard_log=train_log_path
            )   

    else:
        model = ModelClass(
            policy = rl_copp.params_train["policy"], 
            env = rl_copp.env, 
            verbose=True, 
            tensorboard_log=train_log_path
            )   

    # Make a copy of the configuration file for saving the parameters that will be used for this training
    utils.copy_json_with_id(rl_copp.args.params_file, parameters_used_path, rl_copp.file_id)

    logging.warning("Training will start in few seconds. If you want to end it at any time, press 'F' + Enter key, and then 'Y' + Enter key."
                    " It's not recommended to pause and then resume the training, as it will affect the current episode. That said, grab a cup of coffee and enjoy the process ☕️")
    
    time.sleep(8)

    # Save a timestamp of the beggining of the training
    start_time = time.time()

    # Start the training
    
    try:
        if rl_copp.args.verbose ==0:
            model.learn(
                total_timesteps=rl_copp.params_train['total_timesteps'],
                callback=[checkpoint_callback, stop_callback, eval_train_callback, metrics_callback], 
                # log_interval=25, # This is will be also the minimum timesteps to store a data in a tf.events file.
                tb_log_name=f"{rl_copp.args.robot_name}_tflogs"
                )
        else:
            model.learn(
                total_timesteps=rl_copp.params_train['total_timesteps'],
                callback=[checkpoint_callback, stop_callback, eval_train_callback, metrics_callback], 
                # log_interval=25, # This is will be also the minimum timesteps to store a data in a tf.events file.
                tb_log_name=f"{rl_copp.args.robot_name}_tflogs",
                progress_bar = True
                )

        
    except Exception as e:
        traceback.print_exc()
        logging.critical(f"There was an error during the learning process. Exception: {e}")

    # Save a timestamp of the ending of the training
    end_time = time.time()

    # Save the final trained model
    
    logging.info(f"PATH TO SAVE MODEL: {to_save_model_path}")
    model.save(os.path.join(to_save_model_path, f"{model_name}_last"))

    # Parse metrics from tensorboard log and save them in a csv file. Also, we get the metrics of the last row of that csv file
    _, experiment_csv_path = utils.get_output_csv(model_name, training_metrics_path)
    logging.info(f"PATH TO SAVE CSV TRAINIG: {experiment_csv_path}")
    try:
        _, last_metric_row = utils.parse_tensorboard_logs(train_log_file_path, output_csv=experiment_csv_path)
    except:
        last_metric_row = {}
        logging.error("There was an exception while trying to get data from tensorboard log.")
        # TODO MAnage exception

    # Name of the records csv to store the final values of the training experiment.
    records_csv_name = os.path.join(training_metrics_path,"train_records.csv")

    # Get time to converge using the data from the training csv
    try:
        convergence_time, _, _, _ = utils.get_convergence_point (experiment_csv_path, "Steps", convergence_threshold=0.02)
    except Exception as e:
        logging.error(f"No convergence time was found. Exception: {e}")
        convergence_time = 0.0

    # Construct the dictionary with some data to store in the records file
    data_to_store ={
        "Algorithm" : rl_copp.params_train["sb3_algorithm"],
        "Policy" : rl_copp.params_train["policy"],
        "Action time (s)" : rl_copp.params_env["fixed_actime"],
        "Time to converge (h)" : convergence_time,
        **last_metric_row
    }

    # Update the train record.
    utils.update_records_file (records_csv_name, model_name, start_time, end_time, data_to_store)
    
    # Send a FINISH command to the agent
    rl_copp.env.envs[0].unwrapped._commstoagent.stepExpFinished()   # Unwrapped is needed so we can access the attributes of our wrapped env 

    logging.info("Training completed")

    # Remove monitor folder
    if os.path.exists(rl_copp.log_monitor):
        shutil.rmtree(rl_copp.log_monitor)
        logging.info("Monitor removed")
    else:
        logging.error(f"Monitor not found: {rl_copp.log_monitor}")

    ### Close the CoppeliaSim instance
    rl_copp.stop_coppelia_sim()


if __name__ == "__main__":
    main()