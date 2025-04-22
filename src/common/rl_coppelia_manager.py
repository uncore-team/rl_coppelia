"""
Project: Robot Training and Testing RL Algorithms in CoppeliaSim
Author: Adrián Bañuls Arias
Version: 1.0
Date: 2025-03-12
License: GNU General Public License v3.0

This script contains the definition of the `RLCoppeliaManager` class, which is responsible for managing the entire
training and testing process within the CoppeliaSim environment. It supports both single and parallel training sessions 
and provides functionalities for:

    - Creating the custom environment for the robot (using CoppeliaEnv subclasses).
    - Starting the CoppeliaSim simulation and running the scene.
    - Training the model using the stable_baselines3 algorithm.
    - Testing the model to evaluate its performance.
    - Chained training for running multiple training sessions with different configurations.
    - Stopping the simulation once training or testing is complete.

The class includes functions for handling paths, logging, environment setup, and training management, as well as utilities 
for loading parameters and saving the model and results. This class is the core component for managing reinforcement 
learning tasks using CoppeliaSim as the simulation environment.
"""

import logging
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
from common import utils
from common.coppelia_envs import BurgerBotEnv, TurtleBotEnv
from stable_baselines3.common.env_util import make_vec_env

class RLCoppeliaManager():
    def __init__(self, args):
        """
        Manages the interactions with the CoppeliaSim simulation environment for robot training.

        This class handles the setup, training, testing, and stopping of simulations. It interacts with the CoppeliaSim
        environment through different functions for environment creation, simulation startup, training process, testing process,
        and chained training process. Additionally, it manages logging, parameter loading, and saving of models and training 
        results.
        Args:
            args (Namespace): Command-line arguments passed to the script.

        Attributes:
            paths (dict): Paths to various directories such as models, logs, etc.
            robot_name (str): Name of the robot ("turtleBot", "burgerBot", etc.).
            file_id (str): Unique identifier for the current execution, based on saved logs.
            env (VecEnv): The environment used for training, based on the robot type.
            free_comms_port (int): The communication port used for the simulation.
            current_sim (str): The current CoppeliaSim simulation instance.
            params_env (dict): Environment-specific parameters loaded from the configuration file.
            params_train (dict): Training-specific parameters loaded from the configuration file.
            params_test (dict): Testing-specific parameters loaded from the configuration file.
            args (Namespace): Command-line arguments passed to the script.
        """
        super(RLCoppeliaManager, self).__init__()

        self.args = args

        self.base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        if (not hasattr(args, "robot_name") and hasattr(args, "model_name")) or args.robot_name is None:
            self.robot_name = args.model_name.split('_')[0]
            args.robot_name = self.robot_name
        else:
            self.robot_name = args.robot_name

        self.paths = utils.get_robot_paths(self.base_path, self.robot_name)

        # Get the next index for all the files that will be saved during the execution, so we can assign it as an ID to the execution
        self.file_id = utils.get_file_index (args, self.paths["tf_logs"], self.robot_name)

        # Initialize loggin config
        if hasattr(args, "robot_name"):     # For training or testing
            utils.logging_config(self.paths["script_logs"], "rl", self.robot_name, self.file_id, 
                                log_level=logging.INFO, save_files = True, verbose = args.verbose)
            
        else:   # Other functions (save, tf_start, etc.)
            utils.logging_config(self.paths["script_logs"], "rl", self.robot_name, self.file_id, 
                                log_level=logging.INFO, save_files = False, verbose = args.verbose)

        # Show possible warnings obtained during the parsing arguments function.
        utils.initial_warnings(self)

        # In train and test cases   #TODO This will not work for auto_trainings or sat_trainings, as they need different params files
        if hasattr(args, "params_file"):
            if args.params_file is None:
                args.params_file = utils.get_params_file(self.paths,self.args)
            self.params_env, self.params_train, self.params_test = utils.load_params(args.params_file)

        self.current_sim = None

        # The next free port to be used for the communication between the agent (CoppeliaSim) and the RL side (Python)
        self.free_comms_port = 49054

        if hasattr(args, "dis_parallel_mode") and not self.args.dis_parallel_mode:  # If the parallel mode is not disabled, then it will search for the next free port
            self.free_comms_port = utils.find_next_free_port(start_port=49054)

        # Temporary folder for storing a tensorboard monitor file during training. This is needed for saving a model 
        # based ion the mean reward obtained during training.
        self.log_monitor = os.path.join(self.base_path, "tmp", self.file_id)


    def create_env(self):
        """
        This function creates a custom environment using the CoppeliaEnv child classes (located in coppelia_envs.py script)
        
        If parallel mode has been selected, it will firstly search for the next free port after the default one (49054). After 
        that, it will create the custom environment depending on the robot name specified by the user, and it will vectorize it.

        Two instances are created: ``env``, for training, and ``env_test`` for the EvalCallback that will evaluate the last model every
        x timesteps.
        """
        
        if self.args.robot_name == "burgerBot":
            self.env = make_vec_env(BurgerBotEnv, n_envs=1, monitor_dir=self.log_monitor,
                            env_kwargs={'params_env': self.params_env, 'comms_port': self.free_comms_port})
            self.env_test = make_vec_env(BurgerBotEnv, n_envs=1, monitor_dir=self.log_monitor,
                            env_kwargs={'params_env': self.params_env, 'comms_port': self.free_comms_port+50})
            
            
        elif self.args.robot_name == "turtleBot":
            self.env = make_vec_env(TurtleBotEnv, n_envs=1, monitor_dir=self.log_monitor,
                            env_kwargs={'params_env': self.params_env, 'comms_port': self.free_comms_port})
            self.env_test = make_vec_env(TurtleBotEnv, n_envs=1, monitor_dir=self.log_monitor,
                            env_kwargs={'params_env': self.params_env, 'comms_port': self.free_comms_port+50})
            
        logging.info(f"Environment for training created: {self.env}. Comms port: {self.free_comms_port}")  
        logging.info(f"Environment for testing created: {self.env_test}. Comms port: {self.free_comms_port+50}")
        
            
        
    def start_coppelia_sim(self):
        """
        Run CoppeliaSim and open the selected scene. It will override the code of the 'Agent_Script' file inside the scene with the
        content of the agent_coppelia_script.py.

        Two different instances are needed, so one will be used for training and the other for evaluating during the EvalCallback
        """
        self.current_sim = utils.start_coppelia_and_simulation(self.base_path, self.args, self.params_env, self.free_comms_port)
        self.current_test_sim = utils.start_coppelia_and_simulation(self.base_path, self.args, self.params_env, self.free_comms_port+50)
        


    def stop_coppelia_sim(self, test_mode = False):
        """
        Check if Coppelia simulations are running and, if so, stops every instance.
        """
        utils.stop_coppelia_simulation(self.current_sim)
        utils.stop_coppelia_simulation(self.current_test_sim)
        

