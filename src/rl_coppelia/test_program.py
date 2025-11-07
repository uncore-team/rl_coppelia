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

    ### Create the environment
    if not args.agent_side:
        rl_copp.create_env()

    ### Start CoppeliaSim instance
    if not args.rl_side:
        rl_copp.start_coppelia_sim("Test")

    if not args.agent_side:
        ### Start communication RL - CoppeliaSim
        rl_copp.start_communication()

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
            rl_copp.params_test["sb3_algorithm"] = utils.get_data_from_training_csv(model_name, train_records_csv_name, "Algorithm")
        except:
            rl_copp.params_test["sb3_algorithm"] = rl_copp.params_train["sb3_algorithm"]

        # Get the training algorithm from the parameters file
        ModelClass = getattr(stable_baselines3, rl_copp.params_test["sb3_algorithm"])

        model = ModelClass.load(rl_copp.args.model_name, rl_copp.env)
        logging.info("Model loaded successfully")

        observation, *_ = rl_copp.env.envs[0].reset()
        logging.info(f"Obervation obtained from environment: {observation}")

        action, _states = model.predict(observation, deterministic=True)
        logging.info(f"Actions received: {action}")

        observation, _, terminated, truncated, info = rl_copp.env.envs[0].step(action)
        logging.info(f"Obervation obtained from environment: {observation}")

        action, _states = model.predict(observation, deterministic=True)
        logging.info(f"Actions received: {action}")

        time.sleep(20)