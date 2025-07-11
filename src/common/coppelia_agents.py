import csv
import logging
import math
import os
import random
import sys
import pandas as pd
from common import utils
from spindecoupler import AgentSide # type: ignore
from socketcomms.comms import BaseCommPoint # type: ignore


class CoppeliaAgent:
    def __init__(self, sim, params_env, paths, file_id, comms_port = 49054) -> None:
        """
        Custom agent for CoppeliaSim simulations of different robots.
        
        This agent interfaces with the RLSide class (from spindecoupler package) to
        receive actions and return observations in a reinforcement learning (RL) setup. The
        environment simulates the robot's movement in response to actions.
        
        Args:
            sim (object): CoppeliaSim simulation object, used to interact with the scene.
            params_env (dict): Dictionary containing environment configuration parameters.
            paths (dict): Dictionary containing necessary filesystem paths for logs, configs, and metrics.
            file_id (str): Unique identifier for the current experiment or training session.
            comms_port (int, optional): Communication port for connecting with the RL system. Defaults to 49054.

        Attributes:
            _control_timestep (float): Timestep used by the simulation engine.
            _rltimestep (float): Timestep used by the RL algorithm to issue new actions.
            _waitingforrlcommands (bool): Whether the agent is waiting for new commands from RL.
            _lastaction (dict): The most recent action executed.
            _lastactiont0_sim (float): Simulation timestamp of the last action start.
            _lastactiont0_wall (float): Wall-clock timestamp of the last action start.
            lat_sim (float): Simulated latency between actions.
            lat_wall (float): Wall-clock latency between actions.
            current_sim_offset_time (float): Elapsed simulated time in the current episode.
            current_wall_offset_time (float): Elapsed real time in the current episode.
            reward (float): Current step reward.
            execute_cmd_vel (bool): Whether to send velocity commands to CoppeliaSim.
            colorID (int): Color identifier for visualizing actions in the simulation.
            robot (object): Handle to the robot model in the CoppeliaSim scene.
            robot_baselink (object): Handle to the base of the robot for position tracking.
            target (object): Handle to the target object.
            laser (object): Handle to the laser scanner sensor.
            generator (object): Handle to the obstacle generator object.
            handle_robot_scripts (object): Script handle for controlling robot motion.
            handle_laser_get_observation_script (object): Script handle for laser observation extraction.
            handle_obstaclegenerators_script (object): Script handle for obstacle generation.
            sim (object): Reference to the CoppeliaSim interface.
            params_env (dict): Environment parameters.
            paths (dict): Filesystem paths used during execution.
            experiment_id (str): Current experiment ID.
            file_id (str): Alias for experiment_id.
            save_scene (bool): Whether to save the current scene configuration.
            scene_configs_path (str): Path to folder containing scene configurations.
            save_scene_csv_folder (str): Output directory for scene CSVs.
            save_traj (bool): Whether to save trajectory information.
            save_trajs_path (str): Directory for saving trajectory CSVs.
            save_traj_csv_folder (str): Directory for the current trajectory output.
            model_ids (list): List of model IDs used in the experiment.
            trajectory (list): List of 2D coordinates forming the robot’s trajectory.
            episode_idx (int): Current episode number.
            action_times (list): Optional list of action durations for each episode.
            scene_to_load_folder (str): Folder name of the scene to load from disk.
            training_started (bool): Flag that is True once training begins.
            finish_rec (bool): True when the FINISH command is received from RL.
            reset_flag (bool): Internal flag to signal that a RESET was requested.
            crash_flag (bool): Internal flag to indicate that a collision occurred.
            first_reset_done (bool): Whether the first reset has been completed.
            _commstoRL (AgentSide): Communication interface with the RL side.
        
        Methods:
            get_observation(): It must be called only by the agent to get an observation from the CoppeliaSim scene.
            reset_simulator(): It must be called only by the agent to restore the simulation in CoppeliaSim scene.
            agent_step(): Executes a step in the agent, waits until the current action is finished or read new instructions from the RL.   
        """
        
        sim.setFloatParam(sim.floatparam_simulation_time_step,0.05)
        self._control_timestep = sim.getSimulationTimeStep()
        self._rltimestep = params_env["fixed_actime"]
        if self._rltimestep <= self._control_timestep:
            raise(ValueError("RL timestep must be > control timestep"))
        self._waitingforrlcommands = True
        self._lastaction = None
        self._lastactiont0_sim = 0.0
        self._lastactiont0_wall = 0.0
        self.lat_sim = 0.0
        self.lat_wall = 0.0
        self.current_sim_offset_time = 0.0
        self.current_wall_offset_time = 0.0

        self.reward = 0
        self.execute_cmd_vel = False
        self.colorID = 1
        
        self.robot = None
        self.robot_baselink = None
        self.target = None
        self.handle_robot_scripts = None

        self.laser = None
        self.generator = None
        
        self.handle_laser_get_observation_script = None
        self.handle_obstaclegenerators_script = None

        self.sim = sim
        self.params_env = params_env

        self.initial_simTime = 0
        self.initial_realTime = 0

        self.paths = paths
        self.first_reset_done = False
        
        # AgentSide doesn't have a timeout, so we do this loop in case that Coppelia scene is executed before the RL.
        while True:
            try:
                logging.info(f"Trying to establish communication using the port {comms_port}")
                self._commstoRL = AgentSide(BaseCommPoint.get_ip(),comms_port)
                break
            except:
                logging.info("Connection with RL failed. Retrying in few secs...")
                sim.wait(20)
        
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
        
    
    def get_observation(self):
        """
        Compute the current distance and angle from the robot to the target.
        
        Returns:
            tuple: distance and angle.
        """

        _, data, _ = self.sim.checkDistance(self.robot_baselink, self.target)
        distance = data[6]
        
        p1 = self.sim.getObjectPose(self.robot_baselink, -1)
        p2 = self.sim.getObjectPose(self.target, -1)
        
        twist = self.sim.getObjectOrientation(self.robot_baselink, -1)
        angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0]) - twist[2]
        
        # Normalize angle to range [-pi, pi]
        if angle > math.pi:
            angle -= 2*math.pi
        elif angle < -math.pi:
            angle += 2*math.pi

        if self.laser is not None:
            lasers_obs=self.sim.callScriptFunction('laser_get_observations',self.handle_laser_get_observation_script)
            return distance, angle, lasers_obs
        
        else:
            return distance, angle
    

    def generate_obs_from_csv(self, row):
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


    def reset_simulator(self):
        """
        Reset the simulator: position the robot and target, and reset the counters.
        If there are obstacles, remove them and create new ones.
        """
        if self.action_times != [] and self.action_times is not None:
            if self.episode_idx < len(self.action_times):
                self._rltimestep = self.action_times[self.episode_idx]
                logging.info(f"Action time set to {self._rltimestep}")
        
        # Set speed to 0. It's important to do this before setting the position and orientation
        # of the robot, to avoid bugs with Coppelia simulation
        self.sim.callScriptFunction('cmd_vel',self.handle_robot_scripts,0,0)
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

                
        # Always remove old obstacles before creating news
        if self.generator is not None:
            # Remove old obstacles
            last_obstacles=self.sim.getObjectsInTree(self.generator,self.sim.handle_all,1) 
            if len(last_obstacles) > 0:
                self.sim.removeObjects(last_obstacles)
        
        # Just place the scene objects at random positions and call 'generate_obs'
        if self.scene_to_load_folder == "" or self.scene_to_load_folder is None:
            # Reset positions and orientation
            current_position = self.sim.getObjectPosition(self.robot_baselink, -1)
            if current_position != [0, 0, 0.06969]:
                random_ori = random.uniform(-math.pi, math.pi)
                self.sim.setObjectPosition(self.robot_baselink, [0, 0, 0.06969],-1)
                self.sim.setObjectOrientation(self.robot_baselink, [0,0,random_ori],-1)

            # Randomize target position
            current_target_position = self.sim.getObjectPosition(self.target, -1)
            if current_target_position != [0, 0, 0]:
                delta_x = random.uniform(-2, 2)
                delta_y = random.uniform(-2, 2)
                self.sim.setObjectPosition(self.target, [delta_x, delta_y, 0], -1)

            if self.generator is not None:
                # Generate new obstacles
                self.sim.callScriptFunction('generate_obs',self.handle_obstaclegenerators_script)
            logging.info("Environment RST done")

        # Load the preconfigured scene
        else:
            if self.episode_idx < len(self.action_times):
                self.id_obstacle = 0

                # CSV path
                csv_folder = os.path.join(self.paths["scene_configs"], self.scene_to_load_folder)  
                scene_path = utils.find_scene_csv_in_dir(csv_folder)
                if not os.path.exists(scene_path):
                    logging.error(f"[ERROR] CSV scene file not found: {scene_path}")
                    sys.exit()

                df = pd.read_csv(scene_path)

                # Get all rows that contain targets
                target_rows = df[df['type'] == 'target'].reset_index(drop=True)
                num_targets = len(target_rows)

                if num_targets == 0:
                    logging.error("No targets found in the scene CSV.")
                    sys.exit()

                # Get what target will be used for each episode
                target_idx = self.episode_idx // (len(self.action_times) // num_targets)
                target_idx = min(target_idx, num_targets - 1)  

                # Initialize target counter
                current_target_idx = 0

                for _, row in df.iterrows():
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
                logging.info(f"Episode {self.episode_idx}: Using target #{target_idx} with position {target_rows.iloc[target_idx][['x','y']].tolist()}")

        # Save current scene configuration for further analysis
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
                if self.laser is not None:
                    distance, angle, laser_obs = self.get_observation()
                    if self.params_env["obs_time"]:
                        observation = {"distance": distance, "angles": angle, "laser_obs0": laser_obs[0],
                                        "laser_obs1": laser_obs[1], "laser_obs2":laser_obs[2], 
                                        "laser_obs3":laser_obs[3], "action_time": self._rltimestep}
                    else:
                        observation = {"distance": distance, "angles": angle, "laser_obs0": laser_obs[0],
                                        "laser_obs1": laser_obs[1], "laser_obs2":laser_obs[2], 
                                        "laser_obs3":laser_obs[3]}
                else:
                    distance, angle = self.get_observation()
                    if self.params_env["obs_time"]:
                        observation = {"distance": distance, "angles": angle, 
                                       "action_time": self._rltimestep}
                    else:
                        observation = {"distance": distance, "angles": angle}
                    
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
            else:
                if self.laser is not None:
                    laser_obs=self.sim.callScriptFunction('laser_get_observations',self.handle_laser_get_observation_script)
                    logging.debug(f"Laser values during movement: {laser_obs}")
                    if (
                        laser_obs[0] < self.params_env["max_crash_dist_critical"] or
                        laser_obs[3] < self.params_env["max_crash_dist_critical"] or
                        any(laser_obs[i] < self.params_env["max_crash_dist"] for i in [1, 2])
                    ) and not self.crash_flag:
                        logging.info("Crash during action execution, episode will finish once the action completes")
                        self.crash_flag = True

            action = self._lastaction
            
        # Execution state --> The action has finished, so it tries to read a new command from RL
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
                    if self.params_env["var_action_time_flag"]:
                        self._rltimestep = action["action_time"]

                    # Get actual time of the last action
                    # if self.episode_start_time == 0:
                    #     self.episode_start_time = self.initial_simTime

                    if self.first_reset_done:
                        self._lastactiont0_sim = 0.0
                        self._lastactiont0_wall = 0.0
                        self.lat_sim = 0.0
                        self.lat_wall = 0.0
                        self.first_reset_done = False
                        

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
                    
                    self._waitingforrlcommands = False # from now on, we are waiting to execute the action
                    self.execute_cmd_vel = True
                    
                # RESET received
                elif rl_instruction[0] == AgentSide.WhatToDo.RESET_SEND_OBS:
                    logging.info("Received: RESET_SEND_OBS")
                    
                    # Reset the simulator: speed to 0, reset robot position and place the target at random position
                    self.reset_simulator()
                    
                    # Get an observation
                    if self.laser is not None:
                        distance, angle, laser_obs = self.get_observation()

                        if self.params_env["obs_time"]:
                            observation = {"distance": distance, "angles": angle, 
                                           "laser_obs0": laser_obs[0], "laser_obs1": laser_obs[1], 
                                           "laser_obs2":laser_obs[2], "laser_obs3":laser_obs[3], 
                                           "action_time": self._rltimestep}
                        else:
                            observation = {"distance": distance, "angles": angle, "laser_obs0": laser_obs[0], "laser_obs1": laser_obs[1], "laser_obs2":laser_obs[2], "laser_obs3":laser_obs[3]}

                    else:
                        distance, angle = self.get_observation()

                        if self.params_env["obs_time"]:
                            observation = {"distance": distance, "angles": angle, 
                                           "action_time": self._rltimestep}
                        else:
                            observation = {"distance": distance, "angles": angle}

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


class BurgerBotAgent(CoppeliaAgent):
    def __init__(self, sim, params_env, paths, file_id, comms_port=49054):
        """
        Custom agent for the BurgerBot robot simulation in CoppeliaSim, inherited from CoppeliaAgent class.

        Args:
            sim: Coppelia object for handling the scene's objects.
            params_env (dict): Dictionary of parameters for configuring the agent.
            comms_port (int, optional): The port to be used for communication with the agent system. Defaults to 49054.
            
        Attributes:
            robot (CoppeliaObject): Robot object in CoppeliaSim scene.
            target (CoppeliaObject): Target object in CoppeliaSim scene.
            handle_robot_scripts (CoppeliaObject): Handle for using the moving the robot in CoppeliaSim scene.
        """
        super(BurgerBotAgent, self).__init__(sim, params_env, paths, file_id, comms_port)

        self.robot = sim.getObject("/Burger")
        self.target = sim.getObject("/Target")
        self.handle_robot_scripts = sim.getScript(1, self.robot)

        logging.info(f"BurgerBot Agent created successfully using port {comms_port}.")


class TurtleBotAgent(CoppeliaAgent):
    def __init__(self, sim, params_env, paths, file_id, comms_port=49054):
        """
        Custom agent for the TurtleBot robot simulation in CoppeliaSim, inherited from CoppeliaAgent class.

        Args:
            sim: Coppelia object for handling the scene's objects.
            params_env (dict): Dictionary of parameters for configuring the agent.
            comms_port (int, optional): The port to be used for communication with the agent system. Defaults to 49054.

        Attributes:
            robot (CoppeliaObject): Robot object in CoppeliaSim scene.
            target (CoppeliaObject): Target object in CoppeliaSim scene.
            robot_baselink (CoppeliaObject): Object of the robot's basein CoppeliaSim scene.
            laser (CoppeliaObject): Lase object in CoppeliaSim scene.
            generator (CoppeliaObject): Obstacles' generator object in CoppeliaSim scene.
            handle_laser_get_observation_script (CoppeliaObject): Handle for using the script which gets the laser observations in CoppeliaSim scene.
            handle_obstaclegenerators_script (CoppeliaObject): Handle for using the script which generates the obstacles in CoppeliaSim scene.
            handle_robot_scripts (CoppeliaObject): Handle for using the moving the robot in CoppeliaSim scene.
        """
        super(TurtleBotAgent, self).__init__(sim, params_env, paths, file_id, comms_port)

        self.robot=sim.getObject('/Turtlebot2')
        self.robot_baselink=sim.getObject('/Turtlebot2/base_link_respondable')
        self.target=sim.getObject("/Target")
        self.laser=sim.getObject('/Turtlebot2/fastHokuyo_ROS2')
        self.generator=sim.getObject('/ObstaclesGenerator')
        self.handle_laser_get_observation_script=sim.getScript(1,self.laser,'laser_get_observations')
        self.handle_obstaclegenerators_script=sim.getScript(1,self.generator,'generate_obstacles')
        self.handle_robot_scripts = sim.getScript(1, self.robot)

        logging.info(f"TurtleBot Agent created successfully using port {comms_port}.")
