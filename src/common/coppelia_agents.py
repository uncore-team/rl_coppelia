import csv
import logging
import math
import os
import random
import sys
from time import sleep, time
from typing import Dict
import pandas as pd
from common import utils
from spindecoupler import AgentSide # type: ignore
from socketcomms.comms import BaseCommPoint # type: ignore


class CoppeliaAgent:

    def _check_variable_timestep(self, params_env: Dict) -> None:
        """
        Check if the action names include 'timestep' to determine if variable timestep is used.
        Sets the _rltimestep and _variable_timestep attributes accordingly.
        Args:
            params_env (dict): Environment parameters loaded from a JSON file.
        """
        # Get list of action names from params
        action_names = params_env.get("action_names", [])

        # Set timestep duration for RL actions
        if "timestep" in action_names:
            # The timestep will be variable, provided in the action dict
            self._rltimestep = None  # placeholder, will be updated per action
            self.variable_timestep = True
            logging.info("Variable timestep detected: will use action value instead of fixed_actime.")
        else:
            # Default to fixed timestep
            self._rltimestep = params_env["fixed_actime"]
            self.variable_timestep = False
            self._validate_rltimestep()
            logging.info(f"Fixed timestep set to {self._rltimestep:.3f} s.")

    
    def _validate_rltimestep (self) -> None:
        """Validate that the RL timestep is strictly greater than control timestep.

        In variable-timestep mode, _rltimestep may be None before the first action;
        in that case the check is skipped.

        Raises:
            ValueError: If _rltimestep is set and not greater than _control_timestep.
        """
        # Skip if not yet set (variable-timestep before first action)
        if getattr(self, "_rltimestep", None) is None:
            return
        if self._rltimestep <= self._control_timestep:
            raise(ValueError("RL timestep must be > control timestep"))


    def _update_rltimestep(self, action: Dict[str, float]) -> None:
        """Update _rltimestep from the incoming action if in variable-timestep mode.

        Args:
            action (dict): Action dictionary received from RL.

        Raises:
            KeyError: If variable mode is enabled but the timestep key is missing.
            ValueError: If the extracted timestep is invalid (<= control timestep).
            TypeError: If the timestep value is not convertible to float.
        """
        if "timestep" not in action:
            raise KeyError("Missing 'timestep' key in action while variable-timestep mode is enabled.")
        
        # Convert and validate type
        val = action["timestep"]
        try:
            self._rltimestep = float(val)
        except (TypeError, ValueError):
            raise TypeError(f"Invalid 'timestep' value: {val!r} (must be numeric)")

        # Check if it is okey in relation with the control timestep of the simulator
        self._validate_rltimestep()


    def __init__(self, sim, params_env, paths, file_id, verbose, comms_port = 49054) -> None:
        """
        Custom agent for CoppeliaSim simulations of different robots.
        
        This agent interfaces with the RLSide class (from spindecoupler package) to
        receive actions and return observations in a reinforcement learning (RL) setup. The
        environment simulates the robot's movement in response to actions.
        Args:
            sim: The CoppeliaSim simulation instance.
            params_env (dict): Environment parameters loaded from a JSON file.
            paths (dict): Dictionary containing various paths for saving/loading data.
            file_id (str): Unique identifier for the current training/testing session.
            verbose (int): Verbosity level for logging.
            comms_port (int, optional): Port for communication with the RL side. Defaults to 49054.
        Attributes:
            _commstoRL: Instance of AgentSide for communication with the RL side.
            _control_timestep (float): Timestep used by the simulation engine.
            _rltimestep (float): Timestep for RL actions.
            _waitingforrlcommands (bool): Flag indicating if the agent is waiting for RL commands.
            _lastaction (dict): The most recent action executed.
            _lastactiont0_sim (float): Simulation time when the last action was received.
            _lastactiont0_wall (float): Wall-clock time when the last action was received.
            lat_sim (float): Last Action Time (LAT) in simulation time.
            lat_wall (float): Last Action Time (LAT) in wall-clock time.
            current_sim_offset_time (float): Elapsed simulated time in the current episode.
            current_wall_offset_time (float): Elapsed wall-clock time in the current episode.
            verbose (int): Verbosity level for logging.
            reward (float): Reward for the current step.
            execute_cmd_vel (bool): Flag to indicate if cmd_vel should be executed.
            colorID (int): Color ID for visualization purposes.
            robot, robot_baselink, target, inner_targer: Handles for robot and target objects in CoppeliaSim.
            distance_line: Handle for the debug line showing distance between robot and target.
            handle_robot_scripts: Handle for robot script in CoppeliaSim.
            laser: Handle for laser sensor object, if any.
            generator: Handle for obstacle generator object, if any.
            handle_laser_get_observation_script: Handle for laser observation script, if any.
            handle_obstaclegenerators_script: Handle for obstacle generator script, if any.
            sim: The CoppeliaSim simulation instance.
            params_env (dict): Environment parameters loaded from a JSON file.
            initial_simTime (float): Initial simulation time when the agent starts its first movement.
            initial_realTime (float): Initial wall-clock time when the agent starts its first movement.
            paths (dict): Dictionary containing various paths for saving/loading data.
            first_reset_done (bool): Flag indicating if the first reset has been done.
            finish_rec (bool): Flag to indicate if Finish flag has been received.
            episode_start_time_sim (float): Simulation time when the current episode started.
            episode_start_time_wall (float): Wall-clock time when the current episode started.
            reset_flag (bool): Flag to indicate if a reset has been requested.
            crash_flag (bool): Flag to indicate if a crash has occurred.
            training_started (bool): Flag to indicate if training has started.
            scene_to_load_folder (str): Folder name for loading preconfigured scenes.
            id_obstacle (int): Counter for obstacle IDs.
            action_times (list): List of action times for loading scenes.
            tuples (list): List of tuples (action_time, target_id) for loading scenes.
            num_targets (int): Number of targets in the loaded scene.
            test_scene_mode (str): Mode for loading test scenes ("alternate_targets" or "alternate_action_times").
            df (pd.DataFrame): DataFrame containing scene configuration loaded from CSV.
            target_rows (pd.DataFrame): DataFrame containing only target rows from the scene configuration.
            save_scene (bool): Flag to indicate if the current scene should be saved.
            scene_configs_path (str): Path to the scene configurations directory.
            experiment_id (str): Unique identifier for the current training/testing session.
            episode_idx (int): Current episode index.
            trajectory (list): List to store the robot's trajectory for the current episode.
            save_scene_csv_folder (str): Folder path for saving scene CSV files.
            save_traj (bool): Flag to indicate if trajectories should be saved.
            model_ids (list): List of model IDs for naming trajectory files.
            save_trajs_path (str): Path to the directory for saving trajectory files.
            save_traj_csv_folder (str): Folder path for saving trajectory CSV files.
        Methods:
            get_observation(): Compute the current distance and angle from the robot to the target.
            get_observation_space(): Returns the observation space of the agent.
            generate_obs_from_csv(row): Generate an obstacle in the CoppeliaSim scene based on a row from the CSV file.
            get_random_object_pos(): Get a random robot/target position inside the container.
            is_position_valid(): Check if a random generated position is valid (no collisions with obstacles)
            reset_simulator(): Reset the simulator: position the robot and target, and reset the counters.
            agent_step(): A step of the agent. Process incoming instructions from the server side and execute actions accordingly.
        
        """

        sim.setFloatParam(sim.floatparam_simulation_time_step,0.05)
        self._control_timestep = sim.getSimulationTimeStep()

        # Set timestep duration for RL actions
        self._check_variable_timestep(params_env)
        
        self._waitingforrlcommands = True
        self._lastaction = None
        self._lastactiont0_sim = 0.0
        self._lastactiont0_wall = 0.0
        self.lat_sim = 0.0
        self.lat_wall = 0.0
        self.current_sim_offset_time = 0.0
        self.current_wall_offset_time = 0.0
        self.verbose = verbose

        self.reward = 0
        self.execute_cmd_vel = False
        self.colorID = 1
        
        self.robot = None
        self.robot_baselink = None
        self.distance_line = None
        self.laser = None

        self.target=sim.getObject("/Target")
        self.inner_target=sim.getObject("/Target/Inner_disk")
        self.container = sim.getObject('/ExternalWall')
        self.generator=sim.getObject('/ObstaclesGenerator')
        self.handle_laser_get_observation_script=None
        self.handle_robot_scripts = None
        self.handle_obstaclegenerators_script=sim.getScript(1,self.generator,'generate_obstacles')        

        self.sim = sim
        self.params_env = params_env

        self.initial_simTime = 0
        self.initial_realTime = 0

        self.paths = paths
        self.first_reset_done = False

        # Communication
        self.comms_port = comms_port
        self._commstoRL = None  # To be initialized externally after agent creation
        
        # Process control variables
        self.finish_rec = False
        self.episode_start_time_sim = 0.0
        self.episode_start_time_wall = 0.0
        self.reset_flag = False
        self.crash_flag = False
        self.training_started = False

        # For loading a scene
        self.scene_to_load_folder = ""
        self.id_obstacle = 0
        self.action_times = []
        self.tuples = []
        self.num_targets = 0
        self.test_scene_mode = ""
        self.df = None
        self.target_rows = None

        # Needed for saving scenes
        self.save_scene = False
        self.scene_configs_path = self.paths["scene_configs"]
        self.experiment_id = file_id
        self.episode_idx = 0
        self.trajectory = []
        self.save_scene_csv_folder = os.path.join(
            self.scene_configs_path,
            self.experiment_id,
            "scene_episode"
        )
        
        # For saving trajectory
        self.save_traj = False
        self.model_ids = []
        self.save_trajs_path = self.paths["testing_metrics"]
        self.save_traj_csv_folder = ""

        # For saving obstacles objects generated
        self.obstacles_objs = None

        # For indicating that the lat reset have been done with the first reset of the scene
        self.lat_reset = False

    
    def start_communication(self):
        while True:
            try:
                logging.info(f"Trying to establish communication using the port {self.comms_port}")
                self._commstoRL = AgentSide(BaseCommPoint.get_ip(),self.comms_port)
                logging.info("Communication with RL established successfully")
                break
            except:
                logging.info("Connection with RL failed. Retrying in few secs...")
                self.sim.wait(20)
        
    
    def get_observation(self):
        """
        Compute the current distance and angle from the robot to the target,
        and draw a debug line showing the measured distance.
        """
        # Check distance between robot and target
        _, data, _ = self.sim.checkDistance(self.robot_baselink, self.inner_target)
        distance = data[6]

        # --- DEBUG: draw line to visualize distance ---
        if self.verbose ==3:
            try:
                # Delete previous debug drawing (if exists)
                if hasattr(self, "distance_line") and self.distance_line is not None:
                    self.sim.removeDrawingObject(self.distance_line)

                # Create a new line-drawing object (size=2, color=yellow)
                self.distance_line = self.sim.addDrawingObject(
                    self.sim.drawing_lines, 2.0, 0.0, -1, 1, [1, 1, 0]
                )

                # Add the two endpoints (world coordinates)
                self.sim.addDrawingObjectItem(self.distance_line, data[0:6])

            except Exception as e:
                logging.info(f"[DEBUG] Could not draw distance line: {e}")

        # --- Compute relative angle ---
        p1 = self.sim.getObjectPose(self.robot_baselink, -1)
        p2 = self.sim.getObjectPose(self.inner_target, -1)
        twist = self.sim.getObjectOrientation(self.robot_baselink, -1)
        angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0]) - twist[2]

        # Normalize angle
        if angle > math.pi:
            angle -= 2 * math.pi
        elif angle < -math.pi:
            angle += 2 * math.pi

        # Laser (optional)
        if self.laser is not None:
            lasers_obs = self.sim.callScriptFunction(
                'laser_get_observations', self.handle_laser_get_observation_script
            )
        else:
            lasers_obs = None

        return distance, angle, lasers_obs


    # def get_observation_space(self):
    #     """
    #     Returns the observation space of the agent.
    #     The observation space includes distance and angle to the target, laser observations (if available), and optionally the action time.
    #     """
    #     observation_space = {}

    #     # Get an observation from the agent
    #     distance, angle, laser_obs = self.get_observation()
    #     # Add all the laser observations "laser_obs{i}"
    #     if laser_obs is not None:
    #         for i, val in enumerate(laser_obs):
    #             observation_space[f"laser_obs{i}"] = val

    #     observation_space = {
    #         "distance": distance,
    #         "angle": angle
    #     }

        

    #     # Add action time to the observation if required    # TODO This belongs to the old version, we need to automatize observation_space creation
    #     # if self.params_env["obs_time"]:
    #     #     observation_space["action_time"] = self._rltimestep

    #     return observation_space


    def get_observation_space(self):
        """Build the observation dict using names from params_env["observation_names"].

        This method calls `get_observation()` (which returns distance, angle, and optionally
        laser observations) and then assembles a dictionary of observations in the exact
        order and naming specified by `self.params_env["observation_names"]`.

        Behavior:
            - Known base signals: "distance", "angle".
            - Laser signals are exposed as "laser_obs{i}" (e.g., laser_obs0, laser_obs1, ...).
            - If lasers are not available but the config includes laser entries, those keys
            will be filled with a default numeric value (0.0) to keep the shape stable.

        Returns:
            dict: Observation dictionary keyed by names in params_env["observation_names"].
        """
        # Get raw measurements from the simulator
        distance, angle, laser_obs = self.get_observation()

        # Build a value pool from available signals
        # (distance, angle are always present; lasers may be None)
        value_pool = {
            "distance": float(distance),
            "angle": float(angle),
        }

        if laser_obs is not None:
            # Expose laser beams as laser_obs0..N-1
            for i, val in enumerate(laser_obs):
                value_pool[f"laser_obs{i}"] = float(val)
        else:
            # If config expects lasers but we don't have them now, fill with defaults.
            # This keeps the observation shape consistent with the Box space.
            logging.warning("No laser observations obtained from get_observation(). They will be set to 0.")
            expected_lasers = int(self.params_env.get("laser_observations", 0))
            for i in range(expected_lasers):
                value_pool[f"laser_obs{i}"] = 0.0

        # Desired names and order come from the params file
        names = self.params_env.get("observation_names")

        if names:
            obs = {}
            for name in names:
                if name in value_pool:
                    obs[name] = value_pool[name]
                else:
                    # Unknown name in config: keep numeric output stable and warn once
                    logging.warning(f"[obs] Unknown observation name in config: '{name}'. Filling 0.0")
                    obs[name] = 0.0
            logging.info(f"Observation stored as: {obs}")
            return obs

        else:
            logging.error("No observation names in params json file. Please check it.")
            raise 

    def generate_obs_from_csv(self, row):
        '''
        Generate an obstacle in the CoppeliaSim scene based on a row from the CSV file.
        Args: 
            row (pd.Series): A row from the CSV file containing 'x' and 'y' coordinates for the obstacle.
        Returns:
            None
        '''
        logging.debug(f"Generating obstacles from csv file")
        height_obstacles = 0.4
        size_obstacles = 0.25

        x, y = row["x"], row["y"]
        logging.debug(f"Placing obstacle at x: {x} and y: {y}")
        obs = self.sim.createPrimitiveShape(5, [size_obstacles, size_obstacles, height_obstacles])
        self.sim.setObjectPosition(obs, self.sim.handle_world, [x, y, height_obstacles / 2])
        self.sim.setObjectAlias(obs, f"Obstacle_csv_{self.id_obstacle}")
        self.sim.setObjectParent(obs, self.generator, True)
        self.sim.setObjectSpecialProperty(obs, self.sim.objectspecialproperty_collidable |
                                        self.sim.objectspecialproperty_measurable |
                                        self.sim.objectspecialproperty_detectable)
        self.sim.setObjectInt32Param(obs, self.sim.shapeintparam_respondable, 1)
        self.sim.setObjectInt32Param(obs, self.sim.shapeintparam_static, 0)
        self.sim.setShapeMass(obs, 1000)
        self.sim.resetDynamicObject(obs)
        
        return


    def get_random_object_pos (self, object_type):
        '''
        Get a random object position inside the container, taking into account the object radius and the container dimensions.
        It is used for locating the robot and the target. 
        
        Args:
            object_type (string): String to indicate if it is the 'robot' or the 'target'.
        Returns:
            tuple: x and y coordinates of the object position.
        '''
        # Get wall info from its customization script
        raw_container = self.sim.readCustomBufferData(self.container, '__config__')
        cfg_wall   = self.sim.unpackTable(raw_container) if raw_container else {}
        containerSideX    = cfg_wall.get('xSize', None)
        containerSideY    = cfg_wall.get('ySize', None)

        if object_type == "target":
            # Get target outer radius from its customization script
            raw_target = self.sim.readCustomBufferData(self.target, '__config__')
            cfg_target = self.sim.unpackTable(raw_target) if raw_target else {}
            objectRadius  = cfg_target.get('outerRadius', None)
        elif object_type == "robot":
            objectRadius = self.params_robot["distance_between_wheels"]/2 + self.params_env["max_crash_dist_critical"] + 0.05 # wheels width aprox

        objectPosX = random.uniform(-containerSideX/2 + objectRadius, containerSideX/2 - objectRadius)
        objectPosY = random.uniform(-containerSideY/2 + objectRadius, containerSideY/2 - objectRadius)

        return objectPosX, objectPosY 
    

    def is_position_valid(self, object_type, posObjectX, posObjectY):
        '''
        Check if the object position is valid (not colliding with any obstacle).
        
        Args:
            object_type (string): String to indicate if it is the 'robot' or the 'target'.
            posObjectX (float): x coordinate of the target position.
            posObjectY (float): y coordinate of the target position.
        Returns:
            bool: True if the position is valid, False otherwise.
        '''
        objs = self.sim.getObjectsInTree(self.obstacles_objs, self.sim.handle_all, 1) or []

        # Calculate the distance between the object proposed position and each obstacle
        for obj in objs:
            pos_obstacle = self.sim.getObjectPosition(obj, self.sim.handle_world)  # [x, y, z]
            dx, dy = (posObjectX - pos_obstacle[0]), (posObjectY - pos_obstacle[1])
            dist = math.hypot(dx, dy)

            # Get the threshold for each case
            if object_type == "robot":
                threshold = self.params_robot["distance_between_wheels"]/2 + self.params_env["max_crash_dist_critical"] + 0.05
            elif object_type == "target":
                threshold = self.params_env["reward_dist_1"]

            # Check if the distance does not respect the minimum threshold
            if dist < threshold:
                return False
        return True


    def reset_simulator(self):
        """
        Reset the simulator: position the robot and target, and reset the counters.
        If there are obstacles, remove them and create new ones.
        """

        # Set speed to 0. It's important to do this before setting the position and orientation
        # of the robot, to avoid bugs with Coppelia simulation
        self.sim.callScriptFunction('cmd_vel',self.handle_robot_scripts,0,0)
        if self.verbose == 3:
            self.sim.callScriptFunction('draw_path', self.handle_robot_scripts, 0,0, self.colorID)

        # Calculate lat:
        self.current_sim_offset_time = self.sim.getSimulationTime()-self.episode_start_time_sim
        self.current_wall_offset_time = self.sim.getSystemTime()-self.episode_start_time_wall

        self.lat_sim= self.current_sim_offset_time-self._lastactiont0_sim
        self.lat_wall= self.current_wall_offset_time-self._lastactiont0_wall

        self._lastactiont0_sim = self.current_sim_offset_time
        self._lastactiont0_wall = self.current_wall_offset_time

        # Reset colorID counter
        self.colorID = 1

        # Save trajectory at the beggining of the reset (last episode traj)
        if self.save_traj:
            if self.trajectory != []:
                if self.model_ids is not None and self.model_ids != []:
                    traj_output_path = os.path.join(self.save_traj_csv_folder, f"trajectory_{self.episode_idx}_{self.model_ids[self.episode_idx-1]}.csv")
                else:
                    traj_output_path = os.path.join(self.save_traj_csv_folder, f"trajectory_{self.episode_idx}.csv")
                with open(traj_output_path, mode='w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=["x", "y"])
                    writer.writeheader()
                    writer.writerows(self.trajectory)
                self.trajectory = []
            
                logging.info(f"Trajectory saved in CSV: {traj_output_path}")

        
        # If 'fixed_obs' flag is not set, remove old obstacles before creating new ones
        if not self.params_env["fixed_obs"]:
            logging.info("Resetting simulator changing obstacles positions")

            # Always remove old obstacles before creating new ones
            if self.generator is not None:
                # Remove old obstacles
                last_obstacles=self.sim.getObjectsInTree(self.generator,self.sim.handle_all,1) 
                if len(last_obstacles) > 0:
                    self.sim.removeObjects(last_obstacles)
        else:
            logging.info("Resetting simulator with fixed obstacles")
        
        # Just place the scene objects at random positions and call 'generate_obs'
        if self.scene_to_load_folder == "" or self.scene_to_load_folder is None:
            # Reset positions and orientation
            current_position = self.sim.getObjectPosition(self.robot_baselink, -1)
            
            # Robot will be always placed at the center of the scene if the obstacles positions 
            # are changing between episodes
            if not self.params_env["fixed_obs"]:
                if current_position != [0, 0, 0.06969]:
                    self.sim.setObjectPosition(self.robot_baselink,-1, [0, 0, 0.06969])
            # Random position for the robot if the obstacles are placed in fixed locations
            else:
                if not self.first_reset_done:
                    self.obstacles_objs = self.sim.callScriptFunction('generate_obs',self.handle_obstaclegenerators_script)

                while True:
                    posX, posY = self.get_random_object_pos('robot')
                    if self.is_position_valid('robot', posX, posY):
                        break
                logging.info(f"Robot new position: {posX}, {posY}")
                self.sim.setObjectPosition(self.robot_baselink,-1, [posX, posY, 0.06969])

            # The orientation will always be randomized
            random_ori = random.uniform(-math.pi, math.pi)
            self.sim.setObjectOrientation(self.robot_baselink,-1,[0,0,random_ori])

            # Randomize target position
            current_target_position = self.sim.getObjectPosition(self.target, -1)
            if current_target_position != [0, 0, 0]:
                if not self.params_env["fixed_obs"]:
                    posX, posY = self.get_random_object_pos('target')
                    self.sim.setObjectPosition(self.target, -1, [posX, posY, 0])
                # As the osbtacles are the same as in the previous episode, we need to check that the
                # new target position is not colliding with any obstacle
                else:
                    while True:
                        posX, posY = self.get_random_object_pos('target')
                        if self.is_position_valid('target', posX, posY):
                            break
                    logging.info(f"Target new position: {posX}, {posY}")
                    self.sim.setObjectPosition(self.target, -1, [posX, posY, 0])

            # If osbtacles are not fixed, generate new ones
            if not self.params_env["fixed_obs"]:
                if self.generator is not None:
                    # Generate new obstacles
                    logging.info(f"Regenerating new obstacles")
                    self.obstacles_objs = self.sim.callScriptFunction('generate_obs',self.handle_obstaclegenerators_script)
            logging.info("Environment RST done")


        # --- LOAD A PRECONFIGURED SCENE FOR TESTING ---
        else:
            if self.episode_idx < len(self.action_times):
                self.id_obstacle = 0

                # Load the CSV file and get all the tuples (action_time, target_id) just once
                if self.episode_idx==0:

                    # CSV path
                    csv_folder = os.path.join(self.paths["scene_configs"], self.scene_to_load_folder)  
                    scene_path = utils.find_scene_csv_in_dir(csv_folder)
                    if not os.path.exists(scene_path):
                        logging.error(f"[ERROR] CSV scene file not found: {scene_path}")
                        sys.exit()

                    self.df = pd.read_csv(scene_path)

                    # Get all rows that contain targets
                    self.target_rows = self.df[self.df['type'] == 'target'].reset_index(drop=True)
                    self.num_targets = len(self.target_rows)

                    if self.num_targets == 0:
                        logging.error("No targets found in the scene CSV.")
                        sys.exit()

                    # Get block size (number of episodes per target)
                    block_size = len(self.action_times) // self.num_targets 

                    # Get unique values of action times
                    unique_times = sorted(set(self.action_times))

                    # Calculate how many times each unique action time is repeated
                    reps = self.action_times.count(unique_times[0]) // self.num_targets

                    # Get a list (tuples) with (action_time, target_id) for each episode
                    # Mode A: alternate targets
                    if self.test_scene_mode == "alternate_targets":
                        logging.info("Test scene mode: alternate_targets")
                        for t in unique_times:
                            for target_id in range(self.num_targets):
                                self.tuples.extend([(t, target_id)] * reps)
                    # Mode B (default): alternate action times
                    else:
                        logging.info("Test scene mode: alternate_action_times")
                        for idx, t in enumerate(self.action_times):
                            target_id = idx // block_size
                            self.tuples.append((t, target_id))
                    logging.debug("Tuples (action_time, target_id) for each episode:", self.tuples)
                # Get what target will be used for each episode
                target_idx = min(self.tuples[self.episode_idx][1], self.num_targets - 1)  

                # Set action time for the episode new episode
                if self.action_times != [] and self.action_times is not None:
                    if self.episode_idx < len(self.action_times):
                        self._rltimestep = self.tuples[self.episode_idx][0]
                        logging.info(f"Action time set to {self._rltimestep}")

                # Initialize target counter
                current_target_idx = 0

                for _, row in self.df.iterrows():
                    x, y = row['x'], row['y']
                    z = 0.06969 if row['type'] == "robot" else 0.0  # Set height for placing the robot

                    if row['type'] == 'robot':
                        self.sim.setObjectPosition(self.robot_baselink, -1, [x, y, z])
                        theta = float(row['theta']) if 'theta' in row and not pd.isna(row['theta']) else 0
                        self.sim.setObjectOrientation(self.robot_baselink, -1, [0, 0, theta])

                    elif row['type'] == 'target':
                        if current_target_idx == target_idx:
                            self.sim.setObjectPosition(self.target, -1, [x, y, 0])
                        current_target_idx += 1

                    elif row['type'] == 'obstacle':
                        self.id_obstacle += 1
                        self.generate_obs_from_csv(row)

                logging.info(f"Scene recreated with {self.id_obstacle} obstacles.")
                logging.info(f"Episode {self.episode_idx}: Using target #{target_idx} with position {self.target_rows.iloc[target_idx][['x','y']].tolist()}")


        # --- SAVE CURRENT SCENE CONFIGURATION FOR FURTHER TESTING ---
        if self.save_scene:            
            # Create list to save all the elements
            scene_elements = []

            # Get and save the position and orientation of the robot
            robot_pos = self.sim.getObjectPosition(self.robot_baselink, -1)
            robot_ori = self.sim.getObjectOrientation(self.robot_baselink, -1)
            scene_elements.append(["robot", robot_pos[0], robot_pos[1], robot_ori[2]]) 

            # Get and save target position
            target_pos = self.sim.getObjectPosition(self.target, -1)
            scene_elements.append(["target", target_pos[0], target_pos[1]])

            # Get obstacles (assuming that they are located under self.generator object in Coppelia scene)
            obstacles = self.sim.getObjectsInTree(self.generator, self.sim.handle_all, 1)
            for obs_handle in obstacles:
                obs_pos = self.sim.getObjectPosition(obs_handle, -1)
                scene_elements.append(["obstacle", obs_pos[0], obs_pos[1]])
            
            csv_path = os.path.join(self.save_scene_csv_folder, f"scene_{self.episode_idx}.csv")
                
            # Save CSV file
            with open(csv_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["type", "x", "y", "theta"])
                writer.writerows(scene_elements)

            logging.info(f"Scene saved in CSV: {csv_path}")

        self.episode_idx = self.episode_idx + 1

        # Set first reset flag to True
        if (self.episode_idx<=1):
            self.first_reset_done = True


    def agent_step(self):
        """
        A step of the agent. Process incoming instructions from the server side and execute actions accordingly.
        Args:
            self: The CoppeliaAgent instance.
        Returns:
            dict: The action to be executed.
        """
        
        action = self._lastaction	# by default, continue executing the same last action

        # Waiting state --> It waits until the current action is finished
        if not self._waitingforrlcommands: # not waiting new commands from RL, just executing last action
            self.current_sim_offset_time = self.sim.getSimulationTime()-self.episode_start_time_sim
            if (self.current_sim_offset_time-self._lastactiont0_sim >= self._rltimestep): # last action finished
            # if (self.sim.getSimulationTime()-self._lastactiont0 >= self._rltimestep):

                logging.info("Act completed.")
                self.execute_cmd_vel = False

                # Get an observation
                observation = self.get_observation_space()
                    
                logging.info(f"Obs send STEP: { {key: round(value, 3) for key, value in observation.items()} }")

                # Send observation to RL
                simTime = self.sim.getSimulationTime() - self.initial_simTime
                self._commstoRL.stepSendObs(observation, simTime, self.crash_flag) # RL was waiting for this; no reward is actually needed here
                self.crash_flag = False  # Reset the flag for next iterations
                self._waitingforrlcommands = True  
            
            # If it's waiting and action is not finished yet, then it will check the laser observations, so the robot will 
            # be aware of collisions evn if it's blind. With this, we avoid the next problem: with large time steps like 5 
            # seconds, the robot can have a collision and then it slides around the obstacle, so when the 5 seconds action 
            # finishes, the robot will be slightly further from the robot, and the collision won't be detected anymore. This
            # is not the desired performance: if there was a collision, then we need to know it.
            
            # If agent is still executing last action
            else:
                if self.laser is not None and not self.crash_flag:
                    # Get laser readings once
                    laser_obs = self.sim.callScriptFunction(
                        'laser_get_observations',
                        self.handle_laser_get_observation_script
                    )
                    logging.debug(f"Laser values during movement: {laser_obs}")

                    # Rename for simplicity
                    p = self.params_env
                    n = int(p["laser_observations"])
                    crit = p["max_crash_dist_critical"]
                    norm = p["max_crash_dist"]

                    crashed = False

                    # 4 laser case: turtlebot -> We have a different distance threshold for lateral measurements
                    if n == 4:
                        # 4-beam layout: 0,3 -> critical; 1,2 -> normal
                        if (
                            laser_obs[0] < crit
                            or laser_obs[3] < crit
                            or any(laser_obs[i] < norm for i in (1, 2))
                        ):
                            crashed = True

                    # Any setup other than 4 beams: burgerbot -> Same distance threshold for all the laser measurements
                    else:
                        if any(d < crit for d in laser_obs):
                            crashed = True

                    if crashed:
                        logging.info("Crash during action execution, episode will finish once the action completes")
                        self.crash_flag = True        
               
            action = self._lastaction
            
        # Execution/ Reading state --> The action has finished, so it tries to read a new command from RL
        else:  # waiting for new RL step() or reset()            
            # read the last (pending) step()/reset() indicator and then proceed accordingly
            rl_instruction = self._commstoRL.readWhatToDo()
            
            if rl_instruction is not None: # otherwise the RL has not sent any new command
                self.training_started = True    # Flag set to True whenever the training starts
            
                # STEP received
                if rl_instruction[0] == AgentSide.WhatToDo.REC_ACTION_SEND_OBS:
                    logging.info("Received: REC_ACTION_SEND_OBS")

                    # Receive an action
                    action = rl_instruction[1]
                    logging.info(f"Action rec: { {key: round(value, 3) for key, value in action.items()} }")
                
                    # Update action time if it's variable
                    # Outdated: if self.params_env["var_action_time_flag"]:
                    # Outdated:    self._rltimestep = action["action_time"]
                    if self.variable_timestep:
                        self._update_rltimestep(action)

                    if self.first_reset_done and not self.lat_reset:
                        self._lastactiont0_sim = 0.0
                        self._lastactiont0_wall = 0.0
                        self.lat_sim = 0.0
                        self.lat_wall = 0.0
                        self.lat_reset = True
                        

                    if self.reset_flag:
                        logging.info(f"LAT sim: {round(self.lat_sim,4)}. LAT wall: {round(self.lat_wall,4)}")
                        self._commstoRL.stepSendLastActDur(self.lat_sim, self.lat_wall)
                        logging.info("LAT already sent")

                        self.episode_start_time_sim = self.sim.getSimulationTime()
                        self.episode_start_time_wall = self.sim.getSystemTime()
                        
                        self.current_sim_offset_time = self.sim.getSimulationTime()-self.episode_start_time_sim
                        self.current_wall_offset_time = self.sim.getSystemTime()-self.episode_start_time_wall

                        self._lastactiont0_sim = self.current_sim_offset_time
                        self._lastactiont0_wall = self.current_wall_offset_time
                        self.reset_flag = False

                    else:
                        self.current_sim_offset_time = self.sim.getSimulationTime()-self.episode_start_time_sim
                        self.current_wall_offset_time = self.sim.getSystemTime()-self.episode_start_time_wall

                        self.lat_sim= self.current_sim_offset_time-self._lastactiont0_sim
                        self.lat_wall= self.current_wall_offset_time-self._lastactiont0_wall

                        self._lastactiont0_sim = self.current_sim_offset_time
                        self._lastactiont0_wall = self.current_wall_offset_time

                        logging.info(f"LAT sim: {round(self.lat_sim,4)}. LAT wall: {round(self.lat_wall,4)}")
                        self._commstoRL.stepSendLastActDur(self.lat_sim, self.lat_wall)
                        logging.info("LAT already sent")
                    self.sim.setFloatSignal('latValueSignal', float(self.lat_sim))
                    logging.info(f"Signal latValueSignal set to {self.lat_sim}")
                        
                    
                    self._waitingforrlcommands = False # from now on, we are waiting to execute the action
                    self.execute_cmd_vel = True
                    
                # RESET received
                elif rl_instruction[0] == AgentSide.WhatToDo.RESET_SEND_OBS:
                    logging.info("Received: RESET_SEND_OBS")
                    
                    # Reset the simulator: speed to 0, reset robot position and place the target at random position
                    self.reset_simulator()
                    
                    # Get an observation
                    observation = self.get_observation_space()

                    logging.info(f"Obs send RESET: { {key: round(value, 3) for key, value in observation.items()} }")
                    
                    # Send the observation and the agent time (simulation time) to the RLSide
                    simTime = self.sim.getSimulationTime() - self.initial_simTime
                    self._commstoRL.resetSendObs(observation, simTime)

                    action = None
                    self.reset_flag = True
                    
                # FINISH received --> the loop ends (train or inference has finished)
                elif rl_instruction[0] == AgentSide.WhatToDo.FINISH:
                    logging.info("Received: FINISH")
                    self.finish_rec = True
                    return {}  # End the loop
                
                # In case another instruction is received, we will get a log
                else:
                    logging.error(f"Received Other: {rl_instruction}")
                    
        self._lastaction = action
        return action  # Continue the loop


# -----------------------------------------------
# -------------- Hardcoded classes --------------
# -----------------------------------------------

class BurgerBotAgent(CoppeliaAgent):
    def __init__(self, sim, params_env, paths, file_id, verbose, comms_port=49054):
        """
        Custom agent for the BurgerBot robot simulation in CoppeliaSim, inherited from CoppeliaAgent class.

        Args:
            sim: Coppelia object for handling the scene's objects.
            params_robot (dict): Dictionary of parameters specific to the robot.
            params_env (dict): Dictionary of parameters for configuring the agent.
            comms_port (int, optional): The port to be used for communication with the agent system. Defaults to 49054.
            
        Attributes:
            robot (CoppeliaObject): Robot object in CoppeliaSim scene.
            robot_baselink (CoppeliaObject): Object of the robot's basein CoppeliaSim scene.
            laser (CoppeliaObject): Lase object in CoppeliaSim scene.
        """
        super(BurgerBotAgent, self).__init__(sim, params_env, paths, file_id, verbose, comms_port)

        self.robot = sim.getObject("/Burger")
        self.robot_baselink = self.robot
        self.laser=sim.getObject('/Burger/Laser')
        self.handle_laser_get_observation_script=sim.getScript(1,self.laser,'laser_get_observations')
        self.handle_robot_scripts = sim.getScript(1, self.robot)

        logging.info(f"BurgerBot Agent created successfully using port {comms_port}.")


class TurtleBotAgent(CoppeliaAgent):
    def __init__(self, sim, params_env, paths, file_id, verbose, comms_port=49054):
        """
        Custom agent for the TurtleBot robot simulation in CoppeliaSim, inherited from CoppeliaAgent class.

        Args:
            sim: Coppelia object for handling the scene's objects.
            params_robot (dict): Dictionary of parameters specific to the robot.
            params_env (dict): Dictionary of parameters for configuring the agent.
            comms_port (int, optional): The port to be used for communication with the agent system. Defaults to 49054.

        Attributes:
            robot (CoppeliaObject): Robot object in CoppeliaSim scene.
            robot_baselink (CoppeliaObject): Object of the robot's basein CoppeliaSim scene.
            laser (CoppeliaObject): Lase object in CoppeliaSim scene.

        """
        super(TurtleBotAgent, self).__init__(sim, params_env, paths, file_id, verbose, comms_port)

        self.robot=sim.getObject('/Turtlebot2')
        self.robot_baselink=sim.getObject('/Turtlebot2/base_link_respondable')
        self.laser=sim.getObject('/Turtlebot2/fastHokuyo_ROS2')
        self.handle_laser_get_observation_script=sim.getScript(1,self.laser,'laser_get_observations')
        self.handle_robot_scripts = sim.getScript(1, self.robot)
        
        logging.info(f"TurtleBot Agent created successfully using port {comms_port}.")
