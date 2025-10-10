import logging
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from abc import ABC, abstractmethod


class CoppeliaEnv(gym.Env, ABC):

    def __init__(self, params_env, comms_port=49054):
        """
        Custom environment for simulation agents in CoppeliaSim, inherited from gym.Env.
        
        This environment interfaces with the AgentSide class (from spindecoupler package) to
        send actions and receive observations in a reinforcement learning setup. The environment 
        simulates the robot's movement in response to linear and angular speed commands using a 
        CoppeliaSim scene.
        
        Args:
            params_env (dict): Dictionary of parameters for configuring the environment
            comms_port (int, optional): The port to be used for communication with the agent system. Defaults to 49054.
        
        Attributes:
            action_space (gym.spaces.Box): Action space of the environment.
            observation_space (gym.spaces.Box): Observation space of the environment.
            _commstoagent (RLSide): Object to interact with the agent system (it will be used by the baseline).
            count (int): Step counter to track the number of steps taken in the episode.
            time_elapsed (float): Time counter to track the duration of the episode.
            n_ep (int): Episode counter.
            terminated (bool): Flag indicating if the current episode reached a terminal state.
            truncated (bool): Flag indicating if the current episode was truncated.
            reward (float): Reward for the current step.
            action_dic (dic): Dictionary of actions to be sent to CoppeliaSim agent.
        
        Methods:
            step(action): Executes a single step in the environment based on the provided action.
            reset(): Resets the environment to its initial state and returns the initial observation.
            _calculate_reward(distance): Calculates the reward based on the current distance.
        """
        super(CoppeliaEnv, self).__init__()

        self._commstoagent = None  # To be initialized externally after environment creation

        # # Open the baseline server on the specified port
        # logging.info(f"Trying to establish communication using the port {comms_port}")
        # self._commstoagent = RLSide(port= comms_port)
        # logging.info(f"Communication opened using port {comms_port}")

        # Extract the parameters for the environment located inside the 'params_file.json' (by default).
        # Make params_env accessible for the methods of the class
        self.params_env = params_env

        # Define action space
        if params_env["finish_episode_flag"]:   # The agent can decide when should the episode finish

            if  not params_env["var_action_time_flag"]:
                self.action_space= spaces.Box(low=np.array([params_env["bottom_lspeed_limit"], params_env["bottom_aspeed_limit"], 0.0],dtype=np.float32), 
                                        high=np.array([params_env["upper_lspeed_limit"],params_env["upper_aspeed_limit"], 1.0],dtype=np.float32), dtype=np.float32)
            else:
                self.action_space= spaces.Box(low=np.array([params_env["bottom_lspeed_limit"], params_env["bottom_aspeed_limit"], 0.0, params_env["bottom_actime_limit"]],dtype=np.float32), 
                                        high=np.array([params_env["upper_lspeed_limit"],params_env["upper_aspeed_limit"], 1.0, params_env["upper_actime_limit"]],dtype=np.float32), dtype=np.float32)
        else:

            if params_env["var_action_time_flag"]:
                self.action_space= spaces.Box(low=np.array([params_env["bottom_lspeed_limit"], params_env["bottom_aspeed_limit"], params_env["bottom_actime_limit"]],dtype=np.float32), 
                                        high=np.array([params_env["upper_lspeed_limit"],params_env["upper_aspeed_limit"], params_env["upper_actime_limit"]],dtype=np.float32), dtype=np.float32)
            else:
                self.action_space= spaces.Box(low=np.array([params_env["bottom_lspeed_limit"], params_env["bottom_aspeed_limit"]],dtype=np.float32), 
                                        high=np.array([params_env["upper_lspeed_limit"],params_env["upper_aspeed_limit"]],dtype=np.float32), dtype=np.float32)

        
        # Number of laser observations
        n_lasers = params_env.get("laser_observations", 4)

        # Define observation space
        if params_env["obs_time"]:
            low = np.array([0, -math.pi] + [0]*n_lasers + [0], dtype=np.float32)
            high = np.array([5, math.pi] + [4]*n_lasers + [100], dtype=np.float32)
        else:
            low = np.array([0, -math.pi] + [0]*n_lasers, dtype=np.float32)
            high = np.array([5, math.pi] + [4]*n_lasers, dtype=np.float32)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # LAT variables
        self.lat_sim = 0
        self.lat_wall = 0
        
        # Process control variables
        self.count=0
        self.time_elapsed=0
        self.total_time_elapsed = 0
        self.n_ep=0
        self.ato = 0
        self.terminated = False
        self.truncated = False
        self.collision_flag = False
        self.max_achieved = False
        self.reward = 0
        self.target_zone = 0
        self.action_dic = {}
        self.tol_lat = 0.3
        self.crash_flag = False
        self.initial_ato=0
        self.reset_flag = False
        self.initial_target_distance = 0

        # Name of the robot handle in CoppeliaSim scene
        self.robot_handle_alias = None  # To be defined by child classes


    @abstractmethod
    def compute_reward(self) -> float:
        """Compute the reward for the given observation, action, and info.
        This method should be implemented by subclasses to define the reward logic.
        """
        ...


    def step(self, action):
        """
        Execute one time step within the environment.

        Args:
            action (np.array): Action vector..

        Returns:
            observation (np.array): The new state.
            reward (float): Reward obtained.
            terminated (bool): Whether the episode has terminated.
            truncated (bool): Whether the episode was truncated.
            info (dict): Additional information.
        """

        logging.info("STEP Call")

        # Get initial simulation time when the first step occurs. We cannot do this inside the reset, because between the reset
        if self.reset_flag:
            self.reset_flag = False
            self.initial_ato = self.ato

        # Make sure that action is a numpy array of 1D, because when testing it can be 2D
        action = action.flatten()   

        # Manage different possibilities: 
        # [linear, angular, finish_flag] / [linear, angular, finish_flag, action_time] / [linear, angular, action_time] / [linear, angular]

        if self.params_env["finish_episode_flag"]:
            if not self.params_env["var_action_time_flag"]:
                self.action_dic = {"linear": action[0],"angular": action[1], "finish_flag": action[2]}
            else:
                self.action_dic = {"linear": action[0],"angular": action[1],"finish_flag": action[2], "action_time": action[3]}
        else:
            if self.params_env["var_action_time_flag"]:
                self.action_dic = {"linear": action[0],"angular": action[1],"action_time": action[2]}
            else:
                self.action_dic = {"linear": action[0],"angular": action[1]}

        # Send action to agent and receive an observation.
        logging.info(f"Send act to agent: { {key: round(value, 3) for key, value in self.action_dic.items()} }.")
        self.lat_sim, self.lat_wall, self.observation, self.crash_flag, self.ato = self._commstoagent.stepSendActGetObs(self.action_dic, timeout = 20.0)
        logging.info(f"Obs rec STEP: { {key: round(value, 3) for key, value in self.observation.items()} }")
        logging.debug(f"REC: crash flag: {self.crash_flag}, ato: {self.ato}")

        # Update counters
        self.count=self.count+1 
        self.time_elapsed=self.ato-self.initial_ato
        self.truncated = False

        # Calculate reward
        self.reward = self.compute_reward()
        logging.info(f"LAT sim: {round(self.lat_sim,4)}. LAT wall: {round(self.lat_wall,4)}. RW: {round(self.reward,4)}")
        if self.lat_sim > (self.params_env["fixed_actime"] + self.tol_lat):
            logging.warning(f"WARNING: LAT is too big for current action time. Lat = {round(self.lat_sim,4)}, A_time = {self.params_env['fixed_actime']}")

        # Update episode
        if self.reward !=0:
            logging.debug(f"Episode {self.n_ep} is finished")
            logging.info(f"Time elapsed (sim time): {round(self.time_elapsed, 3)}")
            self.n_ep=self.n_ep+1
        
        # Observation conversion for consistency
        self.observation = np.array(list(self.observation.values()), dtype=np.float32)

        # Add additional information (optional)
        self.info = {
            "terminated": self.terminated, 
            "truncated": self.truncated, 
            "linear_speed":self.action_dic["linear"],
            "angular_speed":self.action_dic["angular"],
            "lat_sim":self.lat_sim,
            "lat_wall":self.lat_wall
            }

        return self.observation, self.reward, self.terminated, self.truncated, self.info
    

    def get_last_info(self):
        return self.info


    def reset(self, seed=None):
        """
        Reset the environment to an initial state and return the initial observation.

        Args: None

        Returns:
            observation (np.array): The initial state.
            info (dict): Additional information.
        """
        logging.info("RESET Call")

        # Get the initial observation after resetting the environment
        self.observation, self.ato = self._commstoagent.resetGetObs(timeout = 20.0)

        # Reset counters and termination flags
        self.terminated = False
        self.truncated = False 
        self.count=0
        self.total_time_elapsed = self.total_time_elapsed + self.time_elapsed
        self.time_elapsed=0
        self.crash_flag = False
        self.reset_flag = True

        logging.info(f"Obs rec RESET: { {key: round(value, 2) for key, value in self.observation.items()} }")
        
        # Convert observation to numpy array
        self.observation = np.array(list(self.observation.values()), dtype=np.float32)

        # Add additional information (optional)
        self.info = {}

        # SB3 learn method needs a tuple
        return (self.observation, self.info) 
    

    def compute_adjusted_reward(self, max_reward):
        """
        Compute the adjusted reward based on the time elapsed.
        
        The reward is reduced based on how much time has passed, where the maximum 
        reduction is half of the maximum reward. The function scales the reduction 
        linearly from 0 to max_reward/2 as time_elapsed goes from 0 to max_time.
        
        Args: 
            max_reward (float): The maximum reward.
            
        Returns:
            float: The adjusted reward after applying the time-based discount.
        """
        
        # Calculate the discount based on time_elapsed. It scales between 0 and max_reward/2.
        discount = (self.time_elapsed / self.params_env["max_time"]) * (max_reward / 2)
        
        # Ensure the discount does not exceed half of the maximum reward (max_reward/2).
        discount = min(discount, max_reward / 2)
        
        # Adjust the reward by subtracting the calculated discount.
        adjusted_reward = max_reward - discount
        
        return adjusted_reward
    

        
class BurgerBotEnv(CoppeliaEnv):
    def __init__(self, params_env, comms_port=49054):
        """
        Custom environment for the BurgerBot robot simulation in CoppeliaSim, inherited from CoppeliaEnv class.

        Args:
            params_env (dict): Dictionary of parameters for configuring the environment
            comms_port (int, optional): The port to be used for communication with the agent system. Defaults to 49054.

        Attributes:
            observation_space (gym.spaces.Box): Observation space of the environment.
                - distance: [0, 5] m.
                - angle: [-pi,pi] rads.
                - time_elapsed: [0, 100] seconds of time consumed by the agent. Optional.
        """
        super(BurgerBotEnv, self).__init__(params_env, comms_port)

        # Set robot handle alias
        self.robot_handle_alias = '/Burger'

    def compute_reward(self):
        """Compute the reward based on the current observation, action, and info.

        Args: None

        Returns:
            reward (float): The computed reward.
        """
        if "laser_observations" in self.params_env:
            laser_obs = list(self.observation.values())[-self.params_env["laser_observations"]:]
        else:
            laser_obs = list(self.observation.values())[-4:]
        distance = self.observation["distance"]

        if self.crash_flag:
            logging.info("Crashed detected during the movement")
            self.collision_flag = True
            self.terminated=True
            self.target_zone = 0
            return self.params_env["crash_penalty"]

        if self.params_env["finish_episode_flag"]:
            if self.action_dic["finish_flag"]<0.5:
                logging.info("Agent self truncated.")
                self.truncated = True
                self.target_zone = 0
                if distance>self.params_env["dist_thresh_finish_flag"]:
                    return self.params_env["finish_flag_penalty"]
                else:
                    return 0

        if any(d < self.params_env["max_crash_dist_critical"] for d in laser_obs):
            logging.info("Crashed")
            self.collision_flag = True
            self.terminated=True
            self.target_zone = 0
            return self.params_env["crash_penalty"]
        
        else:
            self.collision_flag = False

        if distance < self.params_env["reward_dist_3"]:
            self.target_zone = 3
            self.terminated = True
            return self.compute_adjusted_reward(self.params_env["reward_3"])
        elif distance < self.params_env["reward_dist_2"]:
            self.target_zone = 2
            self.terminated = True
            return self.compute_adjusted_reward(self.params_env["reward_2"])
        elif distance < self.params_env["reward_dist_1"]:
            self.target_zone = 1
            self.terminated = True
            return self.compute_adjusted_reward(self.params_env["reward_1"])
        elif distance > self.params_env["max_dist"] or self.time_elapsed > self.params_env["max_time"]:
            self.terminated = True
            logging.info("Max dist or max time achieved")
            self.max_achieved = True
            self.target_zone = 0
            return self.params_env["overlimit_penalty"]
        else:
            self.terminated = False
            self.truncated = False
            self.max_achieved = False
            self.collision_flag = False
            self.target_zone = 0
            return 0

        


class TurtleBotEnv(CoppeliaEnv):
    def __init__(self, params_env, comms_port=49054):
        """
        Custom environment for the TurtleBot robot simulation in CoppeliaSim, inherited from CoppeliaEnv class.

        Args:
            params_env (dict): Dictionary of parameters for configuring the environment
            comms_port (int, optional): The port to be used for communication with the agent system. Defaults to 49054.

        Attributes:
            observation_space (gym.spaces.Box): Observation space of the environment.
                - distance: [0, 5] m.
                - angle: [-pi,pi] rads.
                - laser_obs: 4 floats in the range [0,4] representing the distance in m. to the closest obstacle.
                - time_elapsed: [0, 100] seconds of time consumed by the agent. Optional.
        """
        super(TurtleBotEnv, self).__init__(params_env, comms_port)
        
        # Set robot handle alias
        self.robot_handle_alias = '/Turtlebot2'

    def compute_reward(self):
        """Compute the reward based on the current observation, action, and info.

        Args: None

        Returns:
            reward (float): The computed reward.
        """
        laser_obs = list(self.observation.values())[-self.params_env["laser_observations"]:]
        distance = self.observation["distance"]

        if self.crash_flag:
            logging.info("Crashed detected during the movement")
            self.collision_flag = True
            self.terminated=True
            self.target_zone = 0
            return self.params_env["crash_penalty"]

        if self.params_env["finish_episode_flag"]:
            if self.action_dic["finish_flag"]<0.5:
                logging.info("Agent self truncated.")
                self.truncated = True
                self.target_zone = 0
                if distance>self.params_env["dist_thresh_finish_flag"]:
                    return self.params_env["finish_flag_penalty"]
                else:
                    return 0

        if (
            laser_obs[0] < self.params_env["max_crash_dist_critical"] or
            laser_obs[3] < self.params_env["max_crash_dist_critical"] or
            any(laser_obs[i] < self.params_env["max_crash_dist"] for i in [1, 2])
            ):
            logging.info("Crashed")
            self.collision_flag = True
            self.terminated=True
            self.target_zone = 0
            return self.params_env["crash_penalty"]
        
        else:
            self.collision_flag = False

        if distance < self.params_env["reward_dist_3"]:
            self.target_zone = 3
            self.terminated = True
            return self.compute_adjusted_reward(self.params_env["reward_3"])
        elif distance < self.params_env["reward_dist_2"]:
            self.target_zone = 2
            self.terminated = True
            return self.compute_adjusted_reward(self.params_env["reward_2"])
        elif distance < self.params_env["reward_dist_1"]:
            self.target_zone = 1
            self.terminated = True
            return self.compute_adjusted_reward(self.params_env["reward_1"])
        elif distance > self.params_env["max_dist"] or self.time_elapsed > self.params_env["max_time"]:
            self.terminated = True
            logging.info("Max dist or max time achieved")
            self.max_achieved = True
            self.target_zone = 0
            return self.params_env["overlimit_penalty"]
        else:
            self.terminated = False
            self.truncated = False
            self.max_achieved = False
            self.collision_flag = False
            self.target_zone = 0
            return 0

      