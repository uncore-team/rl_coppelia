import csv
import datetime
import inspect
import logging
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
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
import traceback


class StopTrainingOnKeypress(BaseCallback): # TODO Check that the confirmation message appear two times
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


def _parse_tensorboard_logs(log_dir, 
                           output_csv, 
                           metrics = ["train/loss", "train/actor_loss", "train/critic_loss", "train/entropy_loss", 
                                      "train/value_loss", "train/approx_kl", "train/clip_fraction", "train/explained_variance",
                                      "train/ent_coef", "train/ent_coef_loss", "train/learning_rate",
                                      "rollout/ep_len_mean", "rollout/ep_rew_mean"]):
    """
    Private method that reads TensorBoard logs from a given directory and saves selected metrics into a CSV file.
    
    The function assumes that the metrics are logged simultaneously so that each metric has the same
    number of events. Each row in the CSV will contain the step, wall_time, and the value for each metric.
    
    Parameters:
        log_dir (str): Directory where TensorBoard event files are located.
        output_csv (str): Path to the CSV file to create (default: "metrics.csv").
        metrics (list of str): List of metric names to extract (default: ["loss", "reward", "entropy", etc.]).
    
    Returns:
        tuple: A tuple (rows, last_row_dict) where 'rows' is a list of rows with the extracted metrics
               and 'last_row_dict' is a dictionary corresponding to the last row.
    """
    try:
        # Initialize the events accumulator
        ea = event_accumulator.EventAccumulator(log_dir)
        ea.Reload()  # Load all the events
    except Exception as e:
        logging.error(f"An error happened while trying to initialize the events accumulator. Error: {e}")
        return []

    # Get the avaliable scalar tags
    available_tags = ea.Tags().get("scalars", [])
    logging.debug(f"Avaliable tags: {available_tags}")

    # Filtrate the avaliable metrics
    metrics_present = [m for m in metrics if m in available_tags]

    # If none of our metrics is in the list of avaliable tags, then we indicate it and finish the function
    if not metrics_present:     # TODO Fix this: right now, if there are no logs in tensorboard, the csv of this training will be not saved
        logging.error("None of the specified metrics are available in the logs.")
        return []

    # Define the CSV header
    headers = ["Step", "Step timestamp"] + metrics

    # Get the events' list for each metric
    events_dict = {}
    for metric in metrics:
        if metric in available_tags:
            events_dict[metric] = ea.Scalars(metric)

    # Determine the number of events to iterate over.
    # For safety, we take the minimum number of events among the available metrics.
    n_events = min(len(events) for events in events_dict.values())

    rows = []
    for i in range(n_events):
        # Get the step and wall_time from the first metric.
        step = events_dict[metrics_present[0]][i].step
        wall_time = events_dict[metrics_present[0]][i].wall_time
        formatted_wall_time = datetime.datetime.fromtimestamp(wall_time).strftime("%Y-%m-%d_%H-%M-%S")

        row_dict = {"Step": step, "Step timestamp": formatted_wall_time}

        for metric in metrics:
            if metric in events_dict:
                row_dict[metric] = events_dict[metric][i].value
            else:
                row_dict[metric] = ''  # Fill with '' if there is no value
        rows.append(row_dict)


    # Write data into the CSV file using DictWriter to guarantee column mapping
    with open(output_csv, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    logging.info(f"Metrics saved to {output_csv}")

    # Create a dictionary for the last row using header as keys. Only the second key-value of the dictionary is removed,
    # because it's the timestamp that we are already getting with 'end_time = _get_current_time()' in train_mode() function.    
    last_row_dict = rows[-1] if rows else {}
    
    return rows, last_row_dict


def main(args):
    """
    Train the model using a custom environment.

    This function creates an environment, starts a CoppeliaSim instance and trains an agent 
    using that environment. Finally, it closes the opened simulation.
    """
    rl_copp = RLCoppeliaManager(args)

    ### Start CoppeliaSim instance
    rl_copp.start_soppelia_sim()

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

    # Callback function to save the model every x timesteps
    to_save_callbacks_path, _ = utils.get_next_model_name(callbacks_path, rl_copp.args.robot_name, rl_copp.file_id, callback_mode=True)
    checkpoint_callback = CheckpointCallback(save_freq=rl_copp.params_train['callback_frequency'], save_path=to_save_callbacks_path, name_prefix=rl_copp.args.robot_name)

    # Callback function for stopping the learning process if a specific key is pressed
    stop_callback = StopTrainingOnKeypress(key="F") 

    # Callback for testing and saving the best model every x timesteps
    # Separate evaluation env
    rl_copp.create_env(test_mode=True)

    # Use deterministic actions for evaluation
    eval_callback = EvalCallback(rl_copp.env_test, best_model_save_path="./logs/",
                                log_path="./logs/", eval_freq=500,
                                deterministic=True, render=False)

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

    logging.warning("Training will start in few senconds. If you want to end it at any time, press 'F' + Enter key, and then 'Y' + Enter key."
                    " It's not recommended to pause and then resume the training, as it will affect the current episode. That said, grab a cup of coffee and enjoy the process ☕️")
    
    time.sleep(8)

    # Save a timestamp of the beggining of the training
    start_time = time.time()

    # Start the training
    
    try:
        if rl_copp.args.verbose ==0:
            model.learn(
                total_timesteps=rl_copp.params_train['total_timesteps'],
                callback=[checkpoint_callback, stop_callback, eval_callback], 
                # log_interval=25, # This is will be also the minimum timesteps to store a data in a tf.events file.
                tb_log_name=f"{rl_copp.args.robot_name}_tflogs"
                )
        else:
            model.learn(
                total_timesteps=rl_copp.params_train['total_timesteps'],
                callback=[checkpoint_callback, stop_callback, eval_callback], 
                # log_interval=25, # This is will be also the minimum timesteps to store a data in a tf.events file.
                tb_log_name=f"{rl_copp.args.robot_name}_tflogs",
                progress_bar = True
                )

        
    except Exception as e:
        traceback.print_exc()
        logging.critical(f"There was an error during the learning process. Exception: {e}")
        # sys.exit()

    # Save a timestamp of the ending of the training
    end_time = time.time()

    # Save the final trained model
    to_save_model_path, model_name = utils.get_next_model_name(models_path, rl_copp.args.robot_name, rl_copp.file_id)
    logging.info(f"PATH TO SAVE MODEL: {to_save_model_path}")
    model.save(to_save_model_path)

    # Parse metrics from tensorboard log and save them in a csv file. Also, we get the metrics of the last row of that csv file
    _, experiment_csv_path = utils.get_output_csv(model_name, training_metrics_path)
    logging.info(f"PATH TO SAVE CSV TRAINIG: {experiment_csv_path}")
    try:
        _, last_metric_row = _parse_tensorboard_logs(train_log_file_path, output_csv=experiment_csv_path)
    except:
        last_metric_row = {}
        logging.error("There was an exception while trying to get data from tensorboard log.")
        # TODO MAnage exception

    # Name of the records csv to store the final values of the training experiment.
    records_csv_name = os.path.join(training_metrics_path,"train_records.csv")

    # Get time to converge using the data from the training csv
    try:
        convergence_time, _, _, _ = utils.get_convergence_time (experiment_csv_path, convergence_threshold=0.01)
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

    ### Close the CoppeliaSim instance
    rl_copp.stop_coppelia_sim()


if __name__ == "__main__":
    main()