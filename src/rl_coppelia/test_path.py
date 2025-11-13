#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_path.py

Probe a "RecordedPath" (Dummy + ctrlPt children) in CoppeliaSim at fixed arc-length steps:
- Build a polyline from ctrlPt* dummies (world coordinates).
- Sample poses (x,y,z,yaw) every `step_m` meters along the polyline.
- Teleport the robot to each sampled pose.
- For each pose, randomize the target N times and:
    * Build an observation [distance, angle, laser_obs0..3] (order can follow params_env["observation_names"])
    * Query a trained SB3 policy with model.predict(obs, deterministic=True)
    * Extract only the "timestep" from the action and record it.
- Save a CSV with all timestep trials per path-point, plus mean and std.

Notes:
- Uses CoppeliaSim ZMQ Remote API. Ensure CoppeliaSim is launched with ZMQ plugin available.
- Relies on your RLCoppeliaManager to start/attach to the simulator, like your test.py.

Author: Adrián + ChatGPT
"""

import os
import csv
import logging
import stable_baselines3
from tqdm.auto import tqdm

from common import utils
from common.rl_coppelia_manager import RLCoppeliaManager


# ----------------------------------------------------------------------
# ------------------------------ Main ---------------------------------
# ----------------------------------------------------------------------

def main(args):
    """Drive the path probe and save per-point timestep distributions to CSV."""

    # --- Start/attach to Coppelia/SB3 as in your test.py ---
    rl_copp = RLCoppeliaManager(args)

    # --- Get robot positions if a map has been provided
    if args.map_png_path:
        logging.info(f"Map provided: {args.map_png_path}.")

        res = utils.build_valid_positions_from_map(
            map_png_path=args.map_png_path,
            m_per_px=0.02013,
            origin_xy=(-10.5, -6.0),
            origin_is_lower_left=False,
            obstacle_threshold=15,
            clearance_m=0.35,      # puedes subir a 0.4
            grid_step_m=0.25,      # parametrizable
            interactive_polygon=True
        )

        # Visual check
        utils.preview_mask_and_positions(args.map_png_path, res)

        # Valid positions (not augmented yet, that will be done inside Coppelia scene)
        rl_copp.base_pos_samples = res["positions_xy"].tolist()
        logging.info(f"{len(rl_copp.base_pos_samples)} positions will be sent to the Agent side to be tested.")

    if not args.agent_side:
        rl_copp.create_env()
    if not args.rl_side:
        rl_copp.start_coppelia_sim("TestPath", path_version=True)

    if not args.agent_side:
        ### Start communication RL - CoppeliaSim
        rl_copp.start_communication()

        models_path = rl_copp.paths["models"]
        testing_path = rl_copp.paths["testing_metrics"]
        training_metrics_path = rl_copp.paths["training_metrics"]

        # BUild whole model name path
        rl_copp.args.model_name = os.path.join(models_path, rl_copp.args.model_name)
        logging.info(f"Model used for the testing {rl_copp.args.model_name}")

        # Assure that the algorithm used for testing a model is the same than the one used for training it
        model_name = os.path.splitext(os.path.basename(rl_copp.args.model_name))[0] # Get the model name from the model file path.
        train_records_csv_name = os.path.join(training_metrics_path,"train_records.csv")    # Name of the train records csv to search the algorithm used
        try:
            rl_copp.params_test["sb3_algorithm"] = utils.get_data_from_training_csv(model_name, train_records_csv_name, "Algorithm")
        except:
            rl_copp.params_test["sb3_algorithm"] = rl_copp.params_train["sb3_algorithm"]
        ModelClass = getattr(stable_baselines3, rl_copp.params_test["sb3_algorithm"])

        # Load model (no VecNormalize assumed; if you used it, tell me and we add it)
        model = ModelClass.load(rl_copp.args.model_name, rl_copp.env)

        # Output CSV
        testing_folder = os.path.join(testing_path, f"{model_name}_testing")
        os.makedirs(testing_folder, exist_ok=True)

        experiment_csv_name, experiment_csv_path= utils.get_output_csv(model_name, testing_folder, "path_data")
    
        # CSV header
        observation_names = rl_copp.env.envs[0].unwrapped.params_env.get("observation_names", [])
        id_headers = ["Position idx"] + ["Scenario idx"]  + ["Trial idx"] 
        position_info_headers = ["Pos X"] + ["Pos Y"] 
        headers = id_headers + position_info_headers + ["Timestep"] + observation_names
        
        if rl_copp.base_pos_samples ==[]:
            n_samples = args.n_samples
        else:
            n_samples = len(rl_copp.base_pos_samples)

        logging.info(
            f" ----- Testing the robot {rl_copp.robot_name} with its model {model_name} -----\n"
            f"          --- Path will be sampled in: {n_samples}. ---\n"
            f"          --- Each sample will be tested with {args.n_extra_poses*2+1} scenarios. ---\n"
            f"          --- Each scenario will be repeated {args.trials_per_sample} times. ---\n")

        with open(experiment_csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

            # Iterate sampled points trials_per_sample times each point
            total_robot_poses = n_samples*(args.n_extra_poses*2+1)

            for position_idx in tqdm(range(n_samples), desc="Testing positions", unit="position"):

                for scenario_idx in range(args.n_extra_poses*2+1):

                    # For each sampled point of the path, test different target scenarios
                    for trial_idx in range(args.trials_per_sample):
                        
                        observation, info_obs = rl_copp.env.envs[0].reset()
                        logging.info(f"Position idx: {position_idx}. Scenario idx: {scenario_idx}. Trial idx: {trial_idx}")

                        # Predict action based on last observation
                        action, _states = model.predict(observation, deterministic=True)

                        # Send a step to the agent jsut to confirm that a new action has been predicted successfully
                        observation, _, terminated, truncated, info = rl_copp.env.envs[0].step(action)
                        
                        # Save the predicted timestep
                        ts_value = float(info["actions"]["timestep"])

                        obs_values = [round(float(v), 4) for v in observation.tolist()]

                        # Format obtained info
                        logging.info(f"Extra info in the osbervation: {info_obs}")
                        info_obs = [info_obs["posX"], info_obs["posY"]]

                        row = [position_idx] + [scenario_idx] + [trial_idx] + info_obs + [ts_value] + obs_values
                        writer.writerow(row)

            logging.info(f"[test_path] Saved results to: {experiment_csv_path}")

        rl_copp.env.envs[0].unwrapped._commstoagent.stepExpFinished()
        logging.info("Testing path has finished")
        
        # Optionally stop the sim here (commented to keep your workflow similar to test.py)
        # sim.stopSimulation()
        # rl_copp.stop_coppelia_sim()


if __name__ == "__main__":
    main()
