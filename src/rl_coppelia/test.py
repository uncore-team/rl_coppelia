"""
Project: Robot Training and Testing RL Algorithms in CoppeliaSim
Author: Adrián Bañuls Arias
Version: 1.0
Date: 2025-03-25
License: GNU General Public License v3.0

Description:
    This script tests a single reinforcement learning model in a CoppeliaSim environment.
    The model is evaluated for a given number of iterations, and various performance metrics
    are collected, saved to CSV files, and used to compute a testing summary.

Usage:
    rl_coppelia test --model_name <model_name> [--robot_name <robot_name>]
                     [--scene_path <path_to_scene_file>] [--save_scene] [--save_traj]
                     [--dis_parallel_mode] [--no_gui] [--params_file <path_to_params_file>]
                     [--iterations <int>] [--timestamp <timestamp>] [--verbose <0|1|2|3>]

Features:
    - Automatically detects and launches a CoppeliaSim instance.
    - Loads a trained model using the correct SB3 algorithm used for training.
    - Tests the model in a preconfigured scenario for a set number of iterations.
    - Collects detailed step-by-step data including LATs and speed during the episode.
    - Calculates and logs key episode metrics like reward, time, distance, and collisions.
    - Discards invalid episodes (e.g., those with only one timestep --> the robot starts above the target).
    - Saves a per-episode CSV with all metrics and a secondary CSV with additional data (LATs, speeds).
    - Computes and stores a final metrics summary including success rates and episode averages.
    - Supports optional saving of robot trajectories and appending results to a global test record.
    - Ensures proper environment and simulator cleanup after testing is complete.
"""


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # Suppress TensorFlow warnings
import csv
import logging
import time
import stable_baselines3
from common import utils
from common.rl_coppelia_manager import RLCoppeliaManager
from tqdm.auto import tqdm


def main(args):
    """
    Test an already trained model using a custom environment.

    This function creates an environment, starts a CoppeliaSim instance and tests an agent 
    using that environment. Finally, it closes the opened simulation.
    """
    rl_copp = RLCoppeliaManager(args)

    ### Start CoppeliaSim instance
    rl_copp.start_coppelia_sim()

    ### Create the environment
    rl_copp.create_env()

    ### Test the model

    # Extract the needed paths for testing
    models_path = rl_copp.paths["models"]
    testing_metrics_path = rl_copp.paths["testing_metrics"]
    training_metrics_path = rl_copp.paths["training_metrics"]

    # Check if a model name was provided by the user
    if rl_copp.args.model_name is None:
        _, rl_copp.args.model_name = utils.get_last_model(models_path)
    else:
        rl_copp.args.model_name = os.path.join(models_path, rl_copp.args.model_name)

    logging.info(f"Model used for the testing {rl_copp.args.model_name}")

    # Assure that the algorithm used for testing a model is the same than the one used for training it
    model_name = os.path.splitext(os.path.basename(rl_copp.args.model_name))[0] # Get the model name from the model file path.
    train_records_csv_name = os.path.join(training_metrics_path,"train_records.csv")    # Name of the train records csv to search the algorithm used
    try:
        rl_copp.params_test["sb3_algorithm"] = utils.get_algorithm_for_model(model_name, train_records_csv_name)
    except:
        rl_copp.params_test["sb3_algorithm"] = rl_copp.params_train["sb3_algorithm"]

    # Get the training algorithm from the parameters file
    ModelClass = getattr(stable_baselines3, rl_copp.params_test["sb3_algorithm"])
    
    # Load the model file using the same algorithm used for training that model
    model = ModelClass.load(rl_copp.args.model_name, rl_copp.env)
    
    # Create a folder for the test results
    testing_folder = os.path.join(testing_metrics_path, f"{model_name}_testing")
    os.makedirs(testing_folder, exist_ok=True)

    # Create a subfolder for trajectories
    trajs_folder = os.path.join(testing_folder, "trajs")
    os.makedirs(trajs_folder, exist_ok=True)

    # Get output csv path
    experiment_csv_name, _, experiment_csv_path, otherdata_csv_path = utils.get_output_csv(model_name, testing_folder, train_flag=False)

    # Save a timestamp of the beggining of the testing
    start_time = time.time()

    # Initialize some lists for calculating the final metrics after the testing process.
    rewards_list = []
    time_reach_targets_list = []
    timesteps_counts_list = []
    terminated_list =[]
    collision_list = []
    max_achieved_list = []
    target_zone_list = []
    episode_distances_list = []

    # Get the number of iterations
    if rl_copp.args.iterations is not None:
        n_iter = rl_copp.args.iterations
    else:
        n_iter = rl_copp.params_test['testing_iterations']
    logging.info(f"Running tests for {n_iter} iterations.")

    # Set headers for the different csv files that will be saved
    metrics_headers = [
                'Initial distance (m)', 
                'Reached distance (m)', 
                'Time (s)', 
                'Reward', 
                'Target zone',
                'TimeSteps count', 
                'Terminated', 
                'Truncated', 
                'Crashes',
                'Max limits achieved',
                'Distance traveled (m)'
            ]
    otherdata_headers = ["Episode number", "Linear speed", "Angular speed", "LAT-Sim (s)", "LAT-Wall (s)"]

    # Open a csv file to store the metrics
    with open(experiment_csv_path, mode='w', newline='') as metrics_file:
        metrics_writer = csv.writer(metrics_file)
        metrics_writer.writerow(metrics_headers)

        # Run test x iterations
        # Wrap your range with tqdm to create a progress bar
        for i in tqdm(range(n_iter), desc="Testing Episodes", unit="episode"):
            # The tqdm progress bar will automatically update
            # Get episode number
            n_ep = i+1
            
            # Reset the environment only for the first iteration, as it will be reseted 
            # also after each iteration.
            if i == 0:
                # Get the first observation from the BS3 environment
                observation, *_ = rl_copp.env.envs[0].reset()
            
            # Call init_metrics() for getting the initial time of the iteration
            # and the initial distance to the target
            utils.init_metrics_test(rl_copp.env.envs[0].unwrapped)
            
            # Reset variables to start the iteration
            terminated = False
            truncated = False
            
            # While the simulation doesn't achieve a reward or fail drastically,
            # it will continue trying to get the best reward using the trained model.
            while not (terminated or truncated):
                action, _states = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = rl_copp.env.envs[0].step(action)
                
                # Write speeds of the robot and LATs for each testing step
                try:
                    with open(otherdata_csv_path, mode="r") as f:
                        pass
                except FileNotFoundError:
                    with open(otherdata_csv_path, mode="w", newline='') as f:
                        otherdata_writer = csv.writer(f)
                        otherdata_writer.writerow(otherdata_headers)  # Write the headers

                with open(otherdata_csv_path, mode='a', newline='') as speed_file:
                    otherdata_writer = csv.writer(speed_file)
                    otherdata_writer.writerow([n_ep, info["linear_speed"], info["angular_speed"], info["lat_sim"], info["lat_wall"]])    # saves also the episode number
            
            
            # Call get_metrics(), so we will have the total time of the iteration
            # and the final distance to the targe
            init_target_distance, final_target_distance, time_reach_target, reward_target, timesteps_count, collision_flag, max_achieved, target_zone = utils.get_metrics_test(rl_copp.env.envs[0].unwrapped)
            
            if terminated:
                if reward_target > 0:
                    logging.info(f"Episode terminated with reward {round(reward_target,2)} inside target zone {target_zone}")
                else:
                    logging.info(f"Episode terminated unsuccessfully with reward {round(reward_target,2)}")
            
            # Reset the environment and get an observation
            observation, *_ = rl_copp.env.envs[0].reset()

            if rl_copp.args.save_traj:
                # Traj file should be saved now (it's saved during the reset of the agent), 
                # so we can calculate the distance traveled in the episode
                traj_file = f"trajectory_{i+1}.csv"
                episode_distance = utils.calculate_episode_distance(trajs_folder, traj_file)
            else:
                episode_distance = 0.0

            # Save the metrics in the lists for using them later
            rewards_list.append(reward_target)
            time_reach_targets_list.append(time_reach_target)
            timesteps_counts_list.append(timesteps_count)
            terminated_list.append(terminated)
            collision_list.append(collision_flag)
            max_achieved_list.append(max_achieved)
            target_zone_list.append(target_zone)
            episode_distances_list.append(episode_distance)
            
            # Write a new row with the metrics in the csv file
            if timesteps_count == 1:    # discard the episode as the robot started on top of the target
                logging.info(f"Episode {n_ep} is discarded, as it has only one timestep.")
            else:
                metrics_writer.writerow([init_target_distance, final_target_distance, time_reach_target, reward_target,
                                        target_zone, timesteps_count, terminated, truncated, collision_flag, max_achieved, 
                                        episode_distance])
            
    logging.info(f"Testing metrics has been saved in {experiment_csv_path}")

    # Save a timestamp of the ending of the testing
    end_time = time.time()  

    time.sleep(0.5)  # Wait a bit to ensure all data is written before closing the files
    
    # Calculate final metrics and save them inside a dic
    avg_reward = sum(rewards_list) / len(rewards_list) if rewards_list else 0
    avg_time_reach_target = sum(time_reach_targets_list) / len(time_reach_targets_list) if time_reach_targets_list else 0
    avg_timesteps_count = sum(timesteps_counts_list) / len(timesteps_counts_list) if timesteps_counts_list else 0
    percentage_max_achieved = (sum(max_achieved_list) / len(max_achieved_list)) * 100 if max_achieved_list else 0
    percentage_collisions = (sum(collision_list) / len(collision_list)) * 100 if collision_list else 0
    percentage_not_finished = percentage_max_achieved + percentage_collisions
    percentage_target_zone_1 = (target_zone_list.count(1) / len(target_zone_list)) * 100
    percentage_target_zone_2 = (target_zone_list.count(2) / len(target_zone_list)) * 100
    percentage_target_zone_3 = (target_zone_list.count(3) / len(target_zone_list)) * 100
    avg_distance_per_episode = sum(episode_distances_list) / len(episode_distances_list) if episode_distances_list else 0

    data_to_store ={
        "Algorithm" : rl_copp.params_test["sb3_algorithm"],
        "Avg reward": avg_reward,
        "Avg time reach target": avg_time_reach_target,
        "Avg timesteps": avg_timesteps_count,
        "Percentage terminated": 100-percentage_not_finished, # As all the episodes are marked as terminated, because if we marked them as truncated,
                                                                # the agent is not picking information from that episode.
        "Percentage truncated": percentage_not_finished,    # So we don't use the truncated flag for calculating them, but the 'max_achieved' and 'collision' 
                                                                # flags, which are triggered when the maximum distance or time are achieved, and when there is 
                                                                # a collision, respectively
        "Number of collisions": sum(collision_list),
        "Target zone 1 (%)": percentage_target_zone_1,
        "Target zone 2 (%)": percentage_target_zone_2,
        "Target zone 3 (%)": percentage_target_zone_3,
        "Average distance per episode (m)": avg_distance_per_episode,
    }

    # Name of the records csv to store the final values of the testing experiment.
    record_csv_name = os.path.join(testing_metrics_path,"test_records.csv")

    # Update the test records file.
    utils.update_records_file (record_csv_name, experiment_csv_name, start_time, end_time, data_to_store)

    # Finish the testing process
    # rl_copp.env.envs[0].reset()
    rl_copp.env.envs[0].unwrapped._commstoagent.stepExpFinished()
    logging.info("Testing has finished")
    
    ### Close the CoppeliaSim instance
    rl_copp.stop_coppelia_sim()


if __name__ == "__main__":
    main()