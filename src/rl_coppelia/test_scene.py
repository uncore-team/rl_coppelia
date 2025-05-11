import csv
import logging
import os
import time
import stable_baselines3
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

    # Set the number of iterations
    n_iter = 1
    logging.info(f"Running tests for only {n_iter} iteration for the preconfigured scene {rl_copp.args.scene_config_path}.")

    # speed_headers = ["Linear speed", "Angular speed"]

    # Run test x iterations
    # Wrap your range with tqdm to create a progress bar
    for i in range(n_iter):
        # Get the observation from the BS3 environment
        observation, *_ = rl_copp.env.envs[0].reset()
    
        # Reset variables to start the iteration
        terminated = False
        truncated = False
        
        # While the simulation doesn't achieve a reward or fail drastically,
        # it will continue trying to get the best reward using the trained model.
        while not (terminated or truncated):
            action, _states = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = rl_copp.env.envs[0].step(action)
            
            # try:
            #     with open(speeds_csv_path, mode="r") as f:
            #         pass
            # except FileNotFoundError:
            #     with open(speeds_csv_path, mode="w", newline='') as f:
            #         speed_writer = csv.writer(f)
            #         speed_writer.writerow(speed_headers)  # Write the headers
            
            # with open(speeds_csv_path, mode='a', newline='') as speed_file:
            #     speed_writer = csv.writer(speed_file)
            #     speed_writer.writerow([info["linear_speed"], info["angular_speed"]])
    

    # Finish the testing process
    rl_copp.env.envs[0].reset()
    rl_copp.env.envs[0].unwrapped._commstoagent.stepExpFinished()
    logging.info(f"Experiment finished")
    

    ### Close the CoppeliaSim instance
    rl_copp.stop_coppelia_sim()


if __name__ == "__main__":
    main()