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
import datetime
import os
import shutil
import sys

from socketcomms.comms import BaseCommPoint
from spindecoupler import RLSide
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
import importlib
import pkgutil
from plugins.envs import get_env_factory
import traceback


class RLCoppeliaManager():

    def _autoload_env_plugins(self) -> None:
        """Autoload env plugins from 'plugins.envs' to populate the registry."""
        src_dir = os.path.join(self.base_path, "src")
        if os.path.isdir(src_dir) and src_dir not in sys.path:
            sys.path.insert(0, src_dir)          # for 'plugins.*'

        if self.base_path not in sys.path:
            sys.path.insert(0, self.base_path)   # for 'envs.*'

        try:
            pkg = importlib.import_module("plugins.envs")
            for finder, name, ispkg in pkgutil.iter_modules(pkg.__path__, "plugins.envs."):
                try:
                    importlib.import_module(name)
                    logging.info(f"[plugins] Imported: {name}")
                except Exception:
                    logging.error(f"[plugins] Failed to import {name}")
                    logging.debug(traceback.format_exc())
        except Exception:
            logging.error(f"Env plugins autoload failed:\n{traceback.format_exc()}")

    # ---------------------- helpers ----------------------

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
    

    def _get_base_path(self) -> str:
        """Return project base path (2 levels up from this file)."""
        return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


    def _resolve_robot_name(self, args) -> str:
        """Resolve robot name from args; fallback to model_name prefix if needed."""
        # If args has robot_name and it is not None, use it
        rn = getattr(args, "robot_name", None)
        if rn:
            return rn

        # Else infer from model_name if available
        model_name = getattr(args, "model_name", None)
        if model_name:
            inferred = model_name.split("_model")[0]
            args.robot_name = inferred  # keep args in sync
            return inferred

        # As a last resort, raise a clear error
        raise ValueError("robot_name could not be resolved: provide --robot_name or a valid --model_name.")


    def _compute_file_id(self, args) -> str:
        """Compute the execution ID depending on the calling script."""
        if self.calling_script != "retrain.py":
            return utils.get_file_index(args, self.paths["tf_logs"], self.robot_name)
        return utils.extract_model_id(self.args.model_name)


    def _configure_logging(self, args) -> None:
        """Configure logging based on whether we save files for this flow."""
        save_files = hasattr(args, "robot_name")  # training/testing flows save logs
        utils.logging_config(
            self.paths["script_logs"],
            "rl",
            self.robot_name,
            self.file_id,
            log_level=logging.INFO,
            save_files=save_files,
            verbose=getattr(args, "verbose", 1),
        )


    def _load_params_if_needed(self, args) -> None:
        """Load params if the flow expects a params file (train/test)."""
        if not hasattr(args, "params_file"):
            return

        if args.params_file is None:
            args.params_file = utils.get_params_file(self.paths, self.args)

        source = os.path.join(self.base_path, "configs", args.params_file)
        if not os.path.isfile(source):
            raise FileNotFoundError(f"Params file not found: {source}")

        self.params_scene, self.params_env, self.params_train, self.params_test = utils.load_params(source)


    def _select_comms_port(self, args, default_start: int) -> int:
        """Return a free comms port; honor --dis_parallel_mode."""
        dis_parallel = getattr(args, "dis_parallel_mode", False)
        if dis_parallel:
            return default_start
        return utils.find_next_free_port(start_port=default_start)


    def _snapshot_pids(self) -> dict:
        """Take a snapshot of current processes (pid -> name)."""
        return {proc.pid: proc.name() for proc in psutil.process_iter(["pid", "name"])}
    

    def _save_notes(self):
        """
        Prompt the user to write notes about the current training experiment and save them into a text file.

        If the 'dis_save_notes' argument is not passed, the user is prompted via terminal to input custom text describing
        the experiment. The notes are then stored in a file named 'experiment_notes.txt' inside the corresponding
        robot folder (robots/<robot_name>). Each new entry is separated by an empty line and includes the experiment ID.

        Example:
            # User types: "Testing PPO with dynamic action times"
            # File robots/burgerBot/experiment_notes.txt will contain:
            #
            # Experiment burgerBot_model_004
            # Testing PPO with dynamic action times
        """
        # Check notes_flag before doing anything else
        dis_notes_flag = getattr(self.args, "dis_save_notes", True)
        if dis_notes_flag:
            return  # Nothing to do if flag is set to True
        
        # Get timestamp
        if hasattr(self.args, "timestamp") and self.args.timestamp is not None:
            timestamp = self.args.timestamp  # Obtained from GUI
        else:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Get process name
        process_name = self.calling_script.split('.')[0]

        # Path to robot folder
        robot_dir = os.path.join(self.base_path, "robots", self.robot_name)

        # Path to notes file
        notes_path = os.path.join(robot_dir, "experiment_notes.txt")

        # Ask user for custom notes
        print("\n📝 Please write your notes for this experiment (press Enter when done):")
        user_notes = input(">> ").strip()

        # Prepare text to append
        entry = f"\n\nExperiment: {self.file_id} - Process: {process_name} - Timestamp: {timestamp}\n{user_notes}\n"

        # TODO Fix filed_id, as for testing functionlity is returning not just the ID but also the timestamp, which is an undesired behavior.

        # Append notes to file
        with open(notes_path, "a", encoding="utf-8") as f:
            f.write(entry)

        logging.info(f"✅ Notes saved successfully in {notes_path}")




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
        self.calling_script = self._get_calling_script()

        # --- Basic setup ---
        self.base_path = self._get_base_path()
        self.robot_name = self._resolve_robot_name(args)
        self.paths = utils.get_robot_paths(self.base_path, self.robot_name)
        self.file_id = self._compute_file_id(args)
        
        # --- Logging & initial warnings ---
        self._configure_logging(args)
        utils.initial_warnings(self)
        self._save_notes()


        # --- Params (train/test flows) --- #TODO This will not work for auto_trainings or sat_trainings, as they need different params files
        self._load_params_if_needed(args)

        # Next steps will be skipped in case of running 'plot.py'
        if self.calling_script != "plot.py":
            # --- Runtime state ---
            self.current_sim = None
            # The next free port to be used for the communication between the agent (CoppeliaSim) and the RL side (Python)
            if args.comms_port is None:
                self.free_comms_port = self._select_comms_port(args, default_start=49054)
            else: 
                self.free_comms_port = args.comms_port
            # Temporary folder for storing a tensorboard monitor file during training. This is needed for saving a model 
            # based ion the mean reward obtained during training.
            self.log_monitor = os.path.join(self.base_path, "tmp", self.file_id)
            # Get current opened processes in the PC so later we can know which ones are the Coppelia new ones.
            self.before_pids = self._snapshot_pids()
            self.current_coppelia_pid = None
            self.terminal_pid = None

            # --- Plugins ---
            self._autoload_env_plugins()


    def create_env(self):
        """Create and vectorize the environment for the selected robot.

        Priority:
        1) If a plugin factory is registered for `self.args.robot_name`, use it.
        2) Fallback to legacy built-ins (burgerBot / turtleBot).

        Returns:
            None. Sets `self.env`.
        """
        # 1) Plugin path (recommended)
        factory = get_env_factory(self.args.robot_name)
        if factory is not None:
            self.env = factory(self)  # factory receives manager, can access params/ports
            logging.info(
                f"[plugins] Environment created via plugin for '{self.args.robot_name}'. "
                f"Comms port: {self.free_comms_port}"
            )
            return

        # 2) Hardcoded environments
        if self.args.robot_name == "burgerBot":
            self.env = make_vec_env(
                BurgerBotEnv,
                n_envs=1,
                monitor_dir=self.log_monitor,
                env_kwargs={
                    "params_scene": self.params_scene,
                    "params_env": self.params_env, 
                    "comms_port": self.free_comms_port                },
            )
        elif self.args.robot_name == "turtleBot":
            self.env = make_vec_env(
                TurtleBotEnv,
                n_envs=1,
                monitor_dir=self.log_monitor,
                env_kwargs={
                    "params_scene": self.params_scene,
                    "params_env": self.params_env, 
                    "comms_port": self.free_comms_port
                },
            )
        else:
            raise ValueError(f"Unknown robot name '{self.args.robot_name}' and no plugin found.")

        logging.info(f"Environment created for robot {self.args.robot_name}. IP: {BaseCommPoint.get_ip()}. Comms port: {self.free_comms_port}")
        

    def start_communication(self):
        base_env = utils.unwrap_env(self.env)
        
        # Open the baseline server on the specified port
        logging.info(f"Trying to establish communication using the port {self.free_comms_port}")
        comm = RLSide(port=self.free_comms_port)
        logging.info(f"Communication opened using port {self.free_comms_port}")

        # Link the comms to the environment
        base_env._commstoagent = comm
    
        
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

    
    def remove_tmp_data(self):
        # Remove monitor folder
        if os.path.exists(self.log_monitor):
            shutil.rmtree(self.log_monitor)
            logging.info("Monitor removed")
        else:
            logging.error(f"Monitor not found: {self.log_monitor}")
