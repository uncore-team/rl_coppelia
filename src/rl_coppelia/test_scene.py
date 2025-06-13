import csv
import datetime
import logging
import os
import re
import time
import pandas as pd
import stable_baselines3
from tqdm.auto import tqdm
from common import utils
from common.rl_coppelia_manager import RLCoppeliaManager
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor



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
    training_metrics_path = rl_copp.paths["training_metrics"]
    scene_configs_path = rl_copp.paths["scene_configs"]

    # Get models' names and paths
    models_names_dict, models_paths_dict = utils.get_model_names_and_paths(rl_copp)

    logging.info(f"Running tests for the preconfigured scene located inside {rl_copp.args.scene_to_load_folder}.")
    
    # Get current timestamp so the metrics.csv file will have an unique name
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    scene_path_folder = os.path.join(scene_configs_path, rl_copp.args.scene_to_load_folder)
    scene_path_metrics_csv = os.path.join(scene_path_folder, f"metrics_{timestamp}.csv")

    # Get action times
    action_times= utils.get_fixed_actimes(rl_copp)

    # Set iterations number
    n_iter = len(rl_copp.args.model_ids)

    # Set headers for csv file
    metrics_headers = [
                'Model name',
                'Action time (s)',
                'Target name',
                'Reached distance (m)', 
                'Time (s)', 
                'Reward', 
                'Target zone',
                'Crashes',
                'Distance traveled (m)',
                'Terminated', 
                'Truncated', 
                'Max limits achieved'
            ]
    # Get how many targets are in the scene
    scene_configs_path = rl_copp.paths["scene_configs"]
    scene_path = os.path.join(scene_configs_path, rl_copp.args.scene_to_load_folder)
    scene_path_csv = utils.find_scene_csv_in_dir(scene_path)
    df = pd.read_csv(scene_path_csv)
    num_targets = (df['type'] == 'target').sum()
    num_models_per_target = len(rl_copp.args.model_ids) // num_targets

    # Create the csv file to save the metrics
    with open(scene_path_metrics_csv, mode='w', newline='') as metrics_file:
        metrics_writer = csv.writer(metrics_file)
        metrics_writer.writerow(metrics_headers)

        # Test an episode for each model in the scene
        for i in tqdm(range(n_iter), desc="Testing Episodes", unit="episode"):
            # Get the model id and name
            target_idx = i // num_models_per_target
            target_name = chr(ord('A') + target_idx)
            model_name = models_names_dict[str(rl_copp.args.model_ids[i])]
            model_index = str(rl_copp.args.model_ids[i])
            match = re.search(r'_model_(\d+)', model_name)
            if match:
                model_id = int(match.group(1))

            logging.info(f"Model used for the testing: {model_name}, model id: {model_id}, located inside {models_paths_dict[model_index]}")

            # Assure that the algorithm used for testing a model is the same than the one used for training it
            train_records_csv_name = os.path.join(training_metrics_path,"train_records.csv")    # Name of the train records csv to search the algorithm used
            try:
                rl_copp.params_test["sb3_algorithm"] = utils.get_algorithm_for_model(model_name, train_records_csv_name)
            except:
                rl_copp.params_test["sb3_algorithm"] = rl_copp.params_train["sb3_algorithm"]

            # Get the training algorithm from the parameters file
            ModelClass = getattr(stable_baselines3, rl_copp.params_test["sb3_algorithm"])
            
            # Load the model file using the same algorithm used for training that model
            model = ModelClass.load(models_paths_dict[model_index], rl_copp.env)

            # speed_headers = ["Linear speed", "Angular speed"]

            if i == 0:
                # Get the first observation from the BS3 environment
                observation, *_ = rl_copp.env.envs[0].reset()

            
            # Reset variables to start the iteration
            terminated = False
            truncated = False
            
            # While the simulation doesn't achieve a reward or fail drastically,
            # it will continue trying to get the best reward using the trained model.
            while not (terminated or truncated):
                action, _states = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = rl_copp.env.envs[0].step(action)

            _, final_target_distance, time_reach_target, reward_target, _, collision_flag, max_achieved, target_zone = utils.get_metrics_test(rl_copp.env.envs[0].unwrapped)
                

            # Reset the environment and get an observation
            observation, *_ = rl_copp.env.envs[0].reset()

            if collision_flag:
                episode_distance = 0.0
            else:
                # Traj file should be saved now (it's saved during the reset of the agent), 
                # so we can calculate the distance traveled in the episode
                trajs_folder = os.path.join(scene_path_folder, "trajs")
                traj_file = f"trajectory_{i+1}_{model_id}.csv"
                episode_distance = utils.calculate_episode_distance(trajs_folder, traj_file)
                logging.info(f"Episode distance calculated: {episode_distance} m for traj file {traj_file}")


            metrics_writer.writerow([model_name, action_times[i], target_name, final_target_distance, 
                                    time_reach_target, reward_target, target_zone, collision_flag,
                                    episode_distance, terminated, truncated, max_achieved])


    # Finish the testing process
    rl_copp.env.envs[0].unwrapped._commstoagent.stepExpFinished()
    logging.info(f"Experiment finished")
    
    ### Close the CoppeliaSim instance
    rl_copp.stop_coppelia_sim()

    # Create a summary CSV file

    # Read the metrics just written
    df_metrics = pd.read_csv(scene_path_metrics_csv)

    # Cast logical fields
    df_metrics['Crashes'] = df_metrics['Crashes'].astype(bool)
    if 'Terminated' in df_metrics.columns:
        df_metrics['Terminated'] = df_metrics['Terminated'].astype(bool)

    # Group by model and target
    grouped = df_metrics.groupby(['Model name', 'Target name'])

    # Build summary rows
    summary_rows = []
    for (model_name, target_name), group in grouped:
        avg_reached_distance = group['Reached distance (m)'].mean()
        avg_time = group['Time (s)'].mean()
        avg_reward = group['Reward'].mean()
        traveled_nonzero = group['Distance traveled (m)'][group['Distance traveled (m)'] > 0]
        avg_traveled = traveled_nonzero.mean() if not traveled_nonzero.empty else 0.0
        crash_rate = group['Crashes'].mean() * 100  
        term_rate = group['Terminated'].mean() * 100 if 'Terminated' in group else None
        zone_counts = group['Target zone'].value_counts(normalize=True) * 100  # en %

        zone_1 = zone_counts.get(1, 0.0)
        zone_2 = zone_counts.get(2, 0.0)
        zone_3 = zone_counts.get(3, 0.0)

        action_time = group['Action time (s)'].iloc[0]

        summary_rows.append([
            model_name, action_time, target_name, avg_reached_distance, avg_time, avg_reward,
            avg_traveled, crash_rate, term_rate, zone_1, zone_2, zone_3
        ])

    # Headers for the summary
    summary_headers = [
        'Model name', 'Action time (s)', 'Target name',
        'Avg reached distance (m)', 'Avg time (s)', 'Avg reward',
        'Avg distance traveled (m)', 'Crash rate (%)', 'Terminated rate (%)',
        'Zone 1 (%)', 'Zone 2 (%)', 'Zone 3 (%)'
    ]

    # Save to CSV
    summary_csv_path = os.path.join(scene_path, "metrics_summary.csv")
    with open(summary_csv_path, mode='w', newline='') as summary_file:
        writer = csv.writer(summary_file)
        writer.writerow(summary_headers)
        writer.writerows(summary_rows)

    logging.info(f"Summary CSV saved: {summary_csv_path}")

    logging.info("Testing process finished successfully.")

if __name__ == "__main__":
    main()