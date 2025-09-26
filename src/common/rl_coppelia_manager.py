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
import os
import shutil
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import inspect
import logging
import psutil
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
from common import utils
from common.coppelia_envs import BurgerBotEnv, TurtleBotEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv
from typing import Callable, Dict
import importlib
import pkgutil


class RLCoppeliaManager():
    _robot_factories: Dict[str, Callable[["RLCoppeliaManager"], VecEnv]] = {}
    @classmethod
    def register_robot(cls, name: str, factory: Callable[["RLCoppeliaManager"], VecEnv]) -> None:
        """Register a new robot factory by name.

        Args:
            name: Unique robot name (e.g., "burgerBot").
            factory: Callable that receives the manager instance and returns a VecEnv.

        Notes:
            Registration is idempotent and overrides existing entries with the same name.
        """
        cls._robot_factories[name] = factory

    def _autoload_robot_plugins(self) -> None:
        """Auto-import robot plugins to populate the registry.

        This will import every module inside 'rl_coppelia.robot_plugins'.
        Each plugin should call `RLCoppeliaManager.register_robot(...)`.

        Safe to call multiple times; imports are cached by Python.
        """
        try:
            pkg_name = "rl_coppelia.robot_plugins"
            pkg = importlib.import_module(pkg_name)
            for m in pkgutil.iter_modules(pkg.__path__, pkg_name + "."):
                importlib.import_module(m.name)
        except Exception as exc:
            logging.debug(f"Robot plugins autoload skipped or failed: {exc}")
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
        self.calling_script = self._get_calling_script()
        self.args = args

        self.base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        if (not hasattr(args, "robot_name") and hasattr(args, "model_name")) or args.robot_name is None:
            self.robot_name = args.model_name.split('_')[0]
            args.robot_name = self.robot_name
        else:
            self.robot_name = args.robot_name

        self.paths = utils.get_robot_paths(self.base_path, self.robot_name)

        # Get the next index for all the files that will be saved during the execution, so we can assign it as an ID to the execution
        if self.calling_script != "retrain.py":
            self.file_id = utils.get_file_index (args, self.paths["tf_logs"], self.robot_name)
        else:
            self.file_id = utils.extract_model_id (self.args.model_name)

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

        # Get current opened processes in the PC so later we can know which ones are the Coppelia new ones.
        self.before_pids = {proc.pid: proc.name() for proc in psutil.process_iter(['pid', 'name'])}

        # Create coppelia current scene process ID and also terminal ID
        self.current_coppelia_pid = None
        self.terminal_pid = None

        # Autoload plugin modules so they can self-register
        self._autoload_robot_plugins()


    def _get_calling_script(self):
        """
        Inspects the call stack to find the first script that is not 'rl_coppelia_manager.py'.
        Returns the base filename (e.g., 'train.py', 'retrain.py').
        """
        for frame in inspect.stack():
            filename = frame.filename
            if filename.endswith('.py') and not filename.endswith('rl_coppelia_manager.py'): 
                return os.path.basename(filename)
        return None


    # def create_env(self):
    #     """
    #     This function creates a custom environment using the CoppeliaEnv child classes (located in coppelia_envs.py script)
        
    #     If parallel mode has been selected, it will firstly search for the next free port after the default one (49054). After 
    #     that, it will create the custom environment depending on the robot name specified by the user, and it will vectorize it.

    #     Two instances are created: ``env``, for training, and ``env_test`` for the EvalCallback that will evaluate the last model every
    #     x timesteps.
    #     """
        
    #     if self.args.robot_name == "burgerBot":
    #         self.env = make_vec_env(BurgerBotEnv, n_envs=1, monitor_dir=self.log_monitor,
    #                         env_kwargs={'params_env': self.params_env, 'comms_port': self.free_comms_port})

            
    #     elif self.args.robot_name == "turtleBot":
    #         self.env = make_vec_env(TurtleBotEnv, n_envs=1, monitor_dir=self.log_monitor,
    #                         env_kwargs={'params_env': self.params_env, 'comms_port': self.free_comms_port})
        
    #     else:   # by default it uses the BurgerBotEnv environment
    #         self.env = make_vec_env(BurgerBotEnv, n_envs=1, monitor_dir=self.log_monitor,
    #                         env_kwargs={'params_env': self.params_env, 'comms_port': self.free_comms_port})
        
            
    #     logging.info(f"Environment for training created: {self.env}. Comms port: {self.free_comms_port}")  

    def create_env(self):
        """Create and vectorize the environment for the selected robot.

        Priority:
        1) If a plugin factory is registered for `self.args.robot_name`, use it.
        2) Fallback to legacy built-ins (burgerBot / turtleBot).
        3) As last resort, use BurgerBotEnv by default.

        Returns:
            None. Sets `self.env`.
        """
        # 1) Plugin path (recommended)
        factory = self._robot_factories.get(self.args.robot_name)
        if factory is not None:
            self.env = factory(self)  # factory receives manager, can access params/ports
            logging.info(
                f"[plugins] Environment created via plugin for '{self.args.robot_name}'. "
                f"Comms port: {self.free_comms_port}"
            )
            return

        # 2) Legacy fallback (keeps current behavior intact)
        if self.args.robot_name == "burgerBot":
            self.env = make_vec_env(
                BurgerBotEnv,
                n_envs=1,
                monitor_dir=self.log_monitor,
                env_kwargs={"params_env": self.params_env, "comms_port": self.free_comms_port},
            )
        elif self.args.robot_name == "turtleBot":
            self.env = make_vec_env(
                TurtleBotEnv,
                n_envs=1,
                monitor_dir=self.log_monitor,
                env_kwargs={"params_env": self.params_env, "comms_port": self.free_comms_port},
            )
        else:
            # 3) Last resort
            self.env = make_vec_env(
                BurgerBotEnv,
                n_envs=1,
                monitor_dir=self.log_monitor,
                env_kwargs={"params_env": self.params_env, "comms_port": self.free_comms_port},
            )

        logging.info(f"Environment for training created: {self.env}. Comms port: {self.free_comms_port}")
        
        
    def start_coppelia_sim(self, process_name:str):
        """
        Run CoppeliaSim and open the selected scene. It will override the code of the 'Agent_Script' file inside the scene with the
        content of the agent_coppelia_script.py.

        Two different instances are needed, so one will be used for training and the other for evaluating during the EvalCallback
        """
        utils.start_coppelia_and_simulation(self, process_name)


    def stop_coppelia_sim(self):
        """
        Check if Coppelia simulations are running and, if so, stops every instance.
        """
        utils.stop_coppelia_simulation(self)

        # Comment/Uncomment if you want to disable/enable CoppeliaSim window auto-closing
        # utils.close_coppelia_sim(self.current_coppelia_pid, self.terminal_pid)

        # Remove monitor folder
        if os.path.exists(self.log_monitor):
            shutil.rmtree(self.log_monitor)
            logging.info("Monitor removed")
        else:
            logging.error(f"Monitor not found: {self.log_monitor}")
