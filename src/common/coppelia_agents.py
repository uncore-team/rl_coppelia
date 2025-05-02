import logging
import math
import random

from spindecoupler import AgentSide # type: ignore
from socketcomms.comms import BaseCommPoint # type: ignore


class CoppeliaAgent:
    def __init__(self, sim, params_env, comms_port = 49054) -> None:
        """
        Custom agent for CoppeliaSim simulations of different robots.
        
        This agent interfaces with the RLSide class (from spindecoupler package) to
        receive actions and return observations in a reinforcement learning (RL) setup. The
        environment simulates the robot's movement in response to actions.
        
        Args:
            sim: Coppelia object for handling the scene's objects.
            params_env (dict): Dictionary of parameters for configuring the agent.
            comms_port (int, optional): The port to be used for communication with the RL system. Defaults to 49054.
        
        Attributes:
            _control_timestep (float): Timestep of the step() cycle in the agent.
            _rltimestep (float): Timestep (or action time) of a RL step (not of an Agent step; the timestep of the Agent step is self._control_timestep)
            _waitingforrlcommands (bool): True if the agent is in waiting state, False if it's in execution state.
            _lastaction (dict): Dictionary that stores the data of the last action.
            _lastactiont0 (float): Timestep of the instant when the last action started.
            reward (float): Reward for the current step, received by the RLSide.
            execute_cmd_vel (bool): Flag for executing or not the cmd_vel script inside CoppeliaSim scene.
            sim (CoppeliaObject): Simulation object from CoppeliaSim.
            robot (CoppeliaObject): Robot object in CoppeliaSim scene.
            target (CoppeliaObject): Target object in CoppeliaSim scene.
            robot_baselink (CoppeliaObject): Object of the robot's basein CoppeliaSim scene.
            laser (CoppeliaObject): Lase object in CoppeliaSim scene.
            generator (CoppeliaObject): Obstacles' generator object in CoppeliaSim scene.
            handle_laser_get_observation_script (CoppeliaObject): Handle for using the script which gets the laser observations in CoppeliaSim scene.
            handle_obstaclegenerators_script (CoppeliaObject): Handle for using the script which generates the obstacles in CoppeliaSim scene.
            handle_robot_scripts (CoppeliaObject): Handle for using the moving the robot in CoppeliaSim scene.
            _commstoRL (AgentSide): Object to interact with the RL side.
            finish_rec (bool): Flag activated when a FINISH command is received from RL side.
            params_env (dict): Dictionary of parameters for configuring the agent.
        
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
        self._lastactiont0 = 0.0
        self.lat = 0.0
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

        # retries = 0
        # MAX_RETRIES = 5
        
        ## AgentSide doesn't have a timeout, so we do this loop in case that Coppelia scene is executed before the RL.
        # while retries < MAX_RETRIES:
        #     try:
        #         logging.info(f"Trying to establish communication using the port {comms_port}")
        #         self._commstoRL = AgentSide(BaseCommPoint.get_ip(),comms_port)
        #         break
        #     except Exception as e:
        #         logging.error(f"Connection with RL failed: {str(e)}. Retrying in few secs...")
        #         retries += 1
        #         if retries >= MAX_RETRIES:
        #             logging.error("Max retries reached. Exiting.")
        #             raise Exception("Failed to establish connection with RL after multiple attempts.")
        #         self.sim.wait(2)
        
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
        self.episode_start_time = 0.0
        self.reset_flag = False
        
    
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
    

    def reset_simulator(self):
        """
        Reset the simulator: position the robot and target, and reset the counters.
        If there are obstacles, remove them and create new ones.
        """
        
        # Set speed to 0. It's important to do this before setting the position and orientation
        # of the robot, to avoid bugs with Coppelia simulation
        self.sim.callScriptFunction('cmd_vel',self.handle_robot_scripts,0,0)
        self.sim.callScriptFunction('draw_path', self.handle_robot_scripts, 0,0, self.colorID)

        # Reset colorID counter
        self.colorID = 1
        
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
            # Remove old obstacles
            last_obstacles=self.sim.getObjectsInTree(self.generator,self.sim.handle_all,1) 
            if len(last_obstacles) > 0:
                self.sim.removeObjects(last_obstacles)

            # Generate new obstacles
            self.sim.callScriptFunction('generate_obs',self.handle_obstaclegenerators_script)
        logging.info("Environment RST done")


    def agent_step(self):
        """
        A step of the agent. Process incoming instructions from the server side and execute actions accordingly.
        """
        
        action = self._lastaction	# by default, continue executing the same last action

        # Waiting state --> It waits until the current action is finished
        if not self._waitingforrlcommands: # not waiting new commands from RL, just executing last action
            self.current_sim_offset_time = self.sim.getSimulationTime()-self.episode_start_time
            if (self.current_sim_offset_time-self._lastactiont0 >= self._rltimestep): # last action finished
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
                self._commstoRL.stepSendObs(observation, None) # RL was waiting for this; no reward is actually needed here
                self._waitingforrlcommands = True
                
            action = self._lastaction
            
        # Execution state --> The action has finished, so it tries to read a new command from RL
        else:  # waiting for new RL step() or reset()
            
            # read the last (pending) step()/reset() indicator and then proceed accordingly
            rl_instruction = self._commstoRL.readWhatToDo()
            
            if rl_instruction is not None: # otherwise the RL has not sent any new command
            
                # STEP received
                if rl_instruction[0] == AgentSide.WhatToDo.REC_ACTION_SEND_OBS:
                    logging.info("Received: REC_ACTION_SEND_OBS")
                    if self.reset_flag:
                        self.episode_start_time = self.sim.getSimulationTime()
                        self._lastactiont0 = 0.0
                        self.reset_flag = False

                    # Receive an action
                    action = rl_instruction[1]
                    logging.info(f"Action rec: { {key: round(value, 3) for key, value in action.items()} }")
                
                    # Update action time if it's variable
                    if self.params_env["var_action_time_flag"]:
                        self._rltimestep = action["action_time"]

                    # Get actual time of the last action
                    self.current_sim_offset_time = self.sim.getSimulationTime()-self.episode_start_time
                    self.lat = self.current_sim_offset_time-self._lastactiont0
                    self._lastactiont0 = self.current_sim_offset_time
                    logging.info(f"LAT: {round(self.lat,4)}")
                    self._waitingforrlcommands = False # from now on, we are waiting to execute the action
                    self.execute_cmd_vel = True
                    self._commstoRL.stepSendLastActDur(self.lat)
                    logging.info("LAT already sent")
            
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
                    
                    # Send the observation to the RLSide
                    self._commstoRL.resetSendObs(obs=observation)

                    action = None
                    
                    # Set reset flag to True, so the simulation time is reset
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
    def __init__(self, sim, params_env, comms_port=49054):
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
        super(BurgerBotAgent, self).__init__(sim, params_env, comms_port)

        self.robot = sim.getObject("/Burger")
        self.target = sim.getObject("/Target")
        self.handle_robot_scripts = sim.getScript(1, self.robot)

        logging.info(f"BurgerBot Agent created successfully using port {comms_port}.")


class TurtleBotAgent(CoppeliaAgent):
    def __init__(self, sim, params_env, comms_port=49054):
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
        super(TurtleBotAgent, self).__init__(sim, params_env, comms_port)

        self.robot=sim.getObject('/Turtlebot2')
        self.robot_baselink=sim.getObject('/Turtlebot2/base_link_respondable')
        self.target=sim.getObject("/Target")
        self.laser=sim.getObject('/Turtlebot2/fastHokuyo_ROS2')
        self.generator=sim.getObject('/ObstaclesGenerator')
        self.handle_laser_get_observation_script=sim.getScript(1,self.laser,'laser_get_observations')
        self.handle_obstaclegenerators_script=sim.getScript(1,self.generator,'generate_obstacles')
        self.handle_robot_scripts = sim.getScript(1, self.robot)

        logging.info(f"TurtleBot Agent created successfully using port {comms_port}.")
