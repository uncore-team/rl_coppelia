""" 
----------------- IMPORTANT: Currently not working properly. -----------------

Project: Robot Training and Testing RL Algorithms in CoppeliaSim
Author: Adrián Bañuls Arias
Version: 1.0
Date: 2025-03-25
License: GNU General Public License v3.0

Description:
    This script retrains an existing reinforcement learning model in a CoppeliaSim environment.
    It resumes training from a previously saved model, extending it with new timesteps while 
    preserving logging, monitoring, and performance tracking features. The script also saves 
    evaluation metrics and model checkpoints throughout the retraining process.

    It uses the same environment and parameters used during the initial training (unless explicitly 
    overridden) and integrates seamlessly with TensorBoard, metrics loggers, and evaluation callbacks.

Usage:
    rl_coppelia retrain --model_name <model_name> --retrain_steps <steps>
                        [--scene_path <path_to_scene>] 
                        [--dis_parallel_mode] [--no_gui]
                        [--params_file <path>] 
                        [--verbose <0|1|2|3>]

Features:
    - Automatically resumes training from a pretrained model checkpoint.
    - Loads and uses the correct algorithm (e.g., SAC, PPO) from training records.
    - Logs new metrics into TensorBoard and saves them in CSV format.
    - Tracks episode count and simulation time for consistency in retraining.
    - Includes customizable callbacks for saving models, early stopping, and metric-based evaluation.
    - Supports verbose logging and intermediate model checkpoint saving.
    - Final metrics are stored in `train_records.csv` with convergence information if available.
    - Cleans up environment and monitor files after training finishes.
"""

import logging
import os
import shutil
import time
import traceback
import stable_baselines3
from common import utils
from stable_baselines3.common.callbacks import CheckpointCallback
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

    ### Create the environment
    rl_copp.create_env()

    ### Start CoppeliaSim instance
    rl_copp.start_coppelia_sim("Retrain")

    ### Start communication RL - CoppeliaSim
    rl_copp.start_communication()

    ### Retrain the model

    # Extract the needed paths for retraining
    models_path = rl_copp.paths["models"]
    training_metrics_path = rl_copp.paths["training_metrics"]
    train_log_path = rl_copp.paths["tf_logs"]
    callbacks_path = rl_copp.paths["callbacks"]
    train_log_file_path = os.path.join(train_log_path,f"{rl_copp.args.robot_name}_tflogs_{rl_copp.file_id}")

    # Get whole path of the model
    model_path_to_load = os.path.join(models_path, rl_copp.args.model_name)

    # Get the model name from the model file path.
    model_name = os.path.splitext(os.path.basename(rl_copp.args.model_name))[0]
    base_model_name = utils.get_base_model_name(model_name)
    

    # Get the algorithm used for the previous training of the model
    train_records_csv_name = os.path.join(training_metrics_path,"train_records.csv")    # Name of the train records csv 
    try:
        retrain_algorithm = utils.get_data_from_training_csv(model_name, train_records_csv_name, "Algorithm")
    except: # SAC by default
        retrain_algorithm = "SAC"

    last_episode = utils.get_data_from_training_csv(base_model_name, train_records_csv_name, "custom/episodes")
    last_sim_time = utils.get_data_from_training_csv(base_model_name, train_records_csv_name, "custom/sim_time")
    

    # Get the training algorithm from the parameters file
    ModelClass = getattr(stable_baselines3, retrain_algorithm)
    
    # Load the model file
    model = ModelClass.load(
        model_path_to_load, 
        env=rl_copp.env,
        tensorboard_log=train_log_file_path
        )

    # Set lat timestep number from last training
    # train_records_csv_name = os.path.join(training_metrics_path,"train_records.csv")    # Name of the train records csv to search the timesteps count
    # model.num_timesteps = utils.get_data_from_training_csv(model_name, train_records_csv_name, column_header="Step")
    # logging.info(f"The timesteps count of the last training of the model {model_name} is {model.num_timesteps} steps.")
    
    model.set_env(rl_copp.env)

    # Get the folder which will contain the final model
    to_save_model_path = os.path.dirname(model_path_to_load)    # From models/turtleBot_model_307/turtleBot_model_307_last --> models/turtleBot_model_307/

    # Callback function to save the model every x timesteps
    to_save_callbacks_path, _ = utils.get_next_model_name(callbacks_path, rl_copp.args.robot_name, rl_copp.file_id, callback_mode=True)
    checkpoint_callback = CheckpointCallback(save_freq=rl_copp.params_train['callback_frequency'], save_path=to_save_callbacks_path, name_prefix=rl_copp.args.robot_name)

    logging.info(f"Model path to load: {model_path_to_load}")
    logging.info(f"Model base name: {base_model_name}")
    logging.info(f"Experiment ID: {rl_copp.file_id}")
    logging.info(f"Robot nmame: {rl_copp.args.robot_name}")
    logging.info(f"to_save_model_path: {to_save_model_path}")
    logging.info(f"to_save_callbacks_path: {to_save_callbacks_path}")
    logging.info(f"Algorithm to be used: {retrain_algorithm}")
    logging.info(f"Last episode: {last_episode}")
    logging.info(f"Last sim time: {last_sim_time}")
    logging.info(f"Tensorboard path: {train_log_file_path}")
    logging.info(f"Retraining mode. Final trained model will be saved in {models_path}")
    
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
    metrics_callback = utils.CustomMetricsCallback(rl_copp, last_episode, last_sim_time)

    # Save a timestamp of the beggining of the training
    start_time = time.time()

    # Start the training
    # Construct tb_log_name
    # tb_log_name = f"{rl_copp.args.robot_name}_tflogs"
    tb_log_name = f"{rl_copp.args.robot_name}_tflogs_{rl_copp.file_id}_retrain"
    logging.info(f"Tensorboard log name: {tb_log_name}")
    try:
        if rl_copp.args.verbose ==0:
            model.learn(
                total_timesteps=rl_copp.args.retrain_steps,
                reset_num_timesteps=False,
                callback=[checkpoint_callback, stop_callback, eval_train_callback, metrics_callback], 
                tb_log_name=tb_log_name
                )
        else:
            model.learn(
                total_timesteps=rl_copp.args.retrain_steps,
                reset_num_timesteps=False,
                callback=[checkpoint_callback, stop_callback, eval_train_callback, metrics_callback], 
                tb_log_name=tb_log_name,
                progress_bar = True
                )
    except Exception as e:
        traceback.print_exc()
        logging.critical(f"There was an error during the learning process. Exception: {e}")

    

    # Save a timestamp of the ending of the training
    end_time = time.time()

    logging.info(f"Training completed in {end_time - start_time:.2f} seconds.")
    logging.info("Waiting 10s for logger to flush...")
    time.sleep(10)  # Allow logger buffer to flush

    # Save the final trained model    
    model_name_to_save = utils.get_next_retrain_model_name(models_path,model_name)

    base_env = utils.get_base_env(rl_copp.env)
    current_episode = base_env.n_ep+float(last_episode)
    current_sim_time = base_env.ato+float(last_sim_time)
    

    try:
        model.logger.record("custom/episodes", current_episode)
        model.logger.record("custom/sim_time", current_sim_time)

        time.sleep(2)
        model.logger.dump(model.num_timesteps)
        logging.info(f"Final custom metrics logged: episode={current_episode}, sim_time={current_sim_time}.")
    except Exception as e:
        logging.warning(f"Logger dump failed: {e}")
    model.save(os.path.join(to_save_model_path, model_name_to_save))
    logging.info(f"Model {model_name_to_save} is saved within {to_save_model_path}")

    # Parse metrics from tensorboard log and save them in a csv file. Also, we get the metrics of the last row of that csv file
    # _last removed from model name

    experiment_csv_path, csv_exists = utils.get_or_create_csv_path(
        base_model_name=base_model_name,
        metrics_folder=training_metrics_path,
        get_output_csv_func=utils.get_output_csv
    )
    if csv_exists:
        logging.info(f"CSV file {experiment_csv_path} already exists. It will be updated with the new data.")
    else:
        logging.info(f"CSV file {experiment_csv_path} does not exist. It will be created with the new data.")

    try:
        # Parse the tensorboard logs and save them in a csv file
        # If the csv file already exists, it will be updated with the new data
        _, last_metric_row = utils.parse_tensorboard_logs(
            log_dir=train_log_file_path,
            output_csv=experiment_csv_path
        )
    except:
        last_metric_row = {}
        logging.error("There was an exception while trying to get data from tensorboard log.")
        # TODO MAnage exception

    # Name of the records csv to store the final values of the training experiment.
    records_csv_name = os.path.join(training_metrics_path,"train_records.csv")

    # Get time to converge using the data from the training csv
    try:
        convergence_time, _, _, _ = utils.get_convergence_point (experiment_csv_path, "SimTime", convergence_threshold=0.05)
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
    utils.update_records_file (records_csv_name, base_model_name, start_time, end_time, data_to_store)
    
    # Send a FINISH command to the agent
    rl_copp.env.envs[0].unwrapped._commstoagent.stepExpFinished()   # Unwrapped is needed so we can access the attributes of our wrapped env 

    logging.info("Training completed")

    # Remove monitor folder
    
    if os.path.exists(rl_copp.log_monitor):
        shutil.rmtree(rl_copp.log_monitor)
        logging.info(f"Monitor file {rl_copp.log_monitor} removed successfully")
    else:
        logging.error(f"Monitor not found: {rl_copp.log_monitor}")

    ### Close the CoppeliaSim instance
    rl_copp.stop_coppelia_sim()