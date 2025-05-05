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
    
    Args:
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


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving the model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    Args:
        check_freq (int): Frequency (in training steps) at which the model should be evaluated during training.
        log_dir (str): Path to the directory where the model will be saved. Must contain the file generated by the ``Monitor`` wrapper.
        verbose (int): Verbosity level.
    """

    def __init__(self, check_freq, save_path, log_dir, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        model_name = os.path.basename(save_path)
        self.save_path = os.path.join(save_path, f"{model_name}_best_train_rw")
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls>10000 and self.n_calls % self.check_freq == 0:
            logging.info("Evaluating model")

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 50 episodes
                mean_reward = np.mean(y[-50:])
                if self.verbose > 0:
                    logging.info(
                        "Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(
                            self.best_mean_reward, mean_reward
                        )
                    )

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Saving best model
                    if self.verbose > 0:
                        logging.info("Saving new best model at {} timesteps".format(x[-1]))
                        logging.info("Saving new best model to {}.zip".format(self.save_path))
                    self.model.save(f"{self.save_path}_{x[-1]}")

        return True


class CustomEvalCallback(EvalCallback):
    """Custom evaluation callback for Stable-Baselines3 with CoppeliaSim integration.

    This callback evaluates the policy every `eval_freq` steps using a separate 
    evaluation environment managed by a custom RL manager. It waits for an episode 
    to finish before performing the evaluation to ensure consistency in environments 
    like CoppeliaSim, where simulation time continues independently.

    Additionally, before saving a new best model (based on evaluation reward), it 
    deletes any previous best model files matching a specific naming pattern 
    (`*_best_test_rw_*.zip`) in the save directory.

    Attributes:
        rl_manager (RLManager): Custom manager containing the evaluation environment.
        steps_since_eval (int): Counter to track steps since the last evaluation.
        pattern (str): Glob pattern to identify old best model files to delete.
    """

    def __init__(self, *args, rl_manager, **kwargs):
        """Initializes the callback and configures model file pattern.

        Args:
            rl_manager (RLCoppeliaManager, optional): Custom manager with the evaluation environment.
            *args: Positional arguments for the EvalCallback.
            **kwargs: Keyword arguments for the EvalCallback.
        """
        super().__init__(*args, **kwargs)
        self.rl_manager = rl_manager
        self.steps_since_eval = 0

        # Pattern to locate old best model files in the save directory
        self.pattern = os.path.join(self.best_model_save_path, "*_best_test_rw_*.zip")

    def _on_step(self) -> bool:
        """Checks whether it's time to evaluate the model and handles the evaluation process.

        Returns:
            bool: True to continue training, False to stop (never stops training in this case).
        """
        self.steps_since_eval += 1

        # Perform evaluation every eval_freq steps
        if (self.num_timesteps > 10000):
            if (self.eval_freq > 0 and self.n_calls % self.eval_freq == 0) or self.steps_since_eval >= self.eval_freq:
                infos = self.locals.get("infos", [])
                terminated = False
                truncated = False

                # Check all environments for termination or truncation (in case of multi-env)
                for idx, info in enumerate(infos):
                    if info.get("terminated", False):
                        terminated = True
                        logging.debug(f"Episode terminated in env {idx}")
                    if info.get("truncated", False):
                        truncated = True
                        logging.debug(f"Episode truncated in env {idx}")

                logging.info(f"terminated: {terminated}, truncated: {truncated}")

                # If episode is still running, then continue the training process until the current episode is finished
                if not (terminated or truncated):
                    logging.info(f"Waiting for episode to finish before doing the evaluation. Steps since last eval: {self.steps_since_eval}")
                    return True
                else:
                    logging.info("Episode is finished, starting the evaluation.")

                # Set evaluation environment
                self.eval_env = self.rl_manager.env_test

                # Evaluate policy over n_eval_episodes
                episode_rewards, _ = evaluate_policy(
                    self.model,
                    self.eval_env,
                    n_eval_episodes=self.n_eval_episodes,
                    render=self.render,
                    deterministic=self.deterministic,
                    return_episode_rewards=True,
                    warn=False,
                )

                mean_reward = sum(episode_rewards) / len(episode_rewards)
                if self.verbose > 0:
                    logging.info(f"Evaluation at step {self.num_timesteps}: mean_reward={mean_reward:.2f}")

                # Save new best model if it outperforms previous best
                if self.best_model_save_path is not None and mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    model_name = os.path.basename(self.best_model_save_path)

                    # Remove any previous best models matching the pattern
                    for file_path in glob.glob(self.pattern):
                        try:
                            os.remove(file_path)
                            logging.info(f"Removed previous best model: {file_path}")
                        except OSError as e:
                            logging.error(f"Error: It was not possible to remove the file {file_path}: {e}")

                    # Save new best model with updated timestep
                    new_model_path = os.path.join(
                        self.best_model_save_path, f"{model_name}_best_test_rw_{self.num_timesteps}"
                    )
                    self.model.save(new_model_path)
                    logging.info(f"New best model saved: {new_model_path}")

                # Log evaluation result to CSV file if logging path is defined
                if self.log_path is not None:
                    log_file_path = os.path.join(self.log_path, "evaluations.csv")
                    with open(log_file_path, "a") as f:
                        f.write(f"{self.num_timesteps},{mean_reward}\n")
                    logging.info(f"Logged evaluation result to {log_file_path}")

                # Reset step counter since last evaluation
                self.steps_since_eval = 0

        return True


class CustomMetricsCallback(BaseCallback):
    def __init__(self, rl_copp, verbose=0):
        super().__init__(verbose)
        self.rl_copp = rl_copp 

    def _on_step(self) -> bool:
        # Log in TensorBoard
        writer = self.logger.writer

        # Get mean reward from tensorboard
        ep_rew_mean = self.logger.name_to_value["rollout/ep_rew_mean"]

        writer.add_scalar("custom/reward_vs_episodes", ep_rew_mean, self.rl_copp.n_ep)
        writer.add_scalar("custom/reward_vs_simtime", ep_rew_mean, self.rl_copp.ato)
        writer.add_scalar("custom/sim_time", self.rl_copp.ato, self.num_timesteps)
        writer.add_scalar("custom/episodes", self.rl_copp.n_ep, self.num_timesteps)


        # self.logger.record("custom/episode_count", self.rl_copp.n_ep)
        # self.logger.record("custom/sim_time", self.rl_copp.ato)

        return True



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
    stop_callback = StopTrainingOnKeypress(key="F") 

    # Callback for saving the best model based on the training reward every x timesteps
    eval_train_callback = SaveOnBestTrainingRewardCallback(
        check_freq=1000, 
        save_path=to_save_model_path, 
        log_dir=rl_copp.log_monitor, 
        verbose=1)

    # Callback for evaluating and saving the best model based on the testing reward every x timesteps
    eval_test_callback = CustomEvalCallback(
        rl_copp.env_test,
        best_model_save_path=to_save_model_path,
        eval_freq=1500,
        deterministic=True,
        render=False,
        rl_manager=rl_copp 
    )

    # Callback for logging custom metrics in Tensorboard
    metrics_callback = CustomMetricsCallback(rl_copp)

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
                callback=[checkpoint_callback, stop_callback, eval_train_callback, eval_test_callback, metrics_callback], 
                # log_interval=25, # This is will be also the minimum timesteps to store a data in a tf.events file.
                tb_log_name=f"{rl_copp.args.robot_name}_tflogs"
                )
        else:
            model.learn(
                total_timesteps=rl_copp.params_train['total_timesteps'],
                callback=[checkpoint_callback, stop_callback, eval_train_callback, eval_test_callback, metrics_callback], 
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
        _, last_metric_row = _parse_tensorboard_logs(train_log_file_path, output_csv=experiment_csv_path)
    except:
        last_metric_row = {}
        logging.error("There was an exception while trying to get data from tensorboard log.")
        # TODO MAnage exception

    # Name of the records csv to store the final values of the training experiment.
    records_csv_name = os.path.join(training_metrics_path,"train_records.csv")

    # Get time to converge using the data from the training csv
    try:
        convergence_time, _, _, _ = utils.get_convergence_point (experiment_csv_path, " Time", convergence_threshold=0.01)
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
    rl_copp.env_test.envs[0].unwrapped._commstoagent.stepExpFinished() 

    logging.info("Training completed")

    # Remove monitor folder
    if os.path.exists(rl_copp.log_monitor):
        shutil.rmtree(rl_copp.log_monitor)

    ### Close the CoppeliaSim instance
    rl_copp.stop_coppelia_sim()


if __name__ == "__main__":
    main()