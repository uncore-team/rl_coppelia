import logging
import math
import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from spindecoupler import RLSide  # type: ignore


class CoppeliaEnv(gym.Env):
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

        # Open the baseline server on the specified port
        logging.info(f"Trying to establish communication using the port {comms_port}")
        self._commstoagent = RLSide(port= comms_port)
        logging.info(f"Communication opened using port {comms_port}")

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

        # Observation space will be defined by child classes, because each robot has different observation spaces
        self.observation_space = None
        
        # Process control variables
        self.count=0
        self.time_elapsed=0
        self.n_ep=0
        self.terminated = False
        self.truncated = False
        self.collision_flag = False
        self.max_achieved = False
        self.reward = 0
        self.action_dic = {}
        self.tol_lat = 0.3
        self.crash_flag = False


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
        self.lat, self.observation, self.crash_flag, _ato = self._commstoagent.stepSendActGetObs(self.action_dic, timeout = 20.0)
        logging.info(f"Obs rec STEP: { {key: round(value, 3) for key, value in self.observation.items()} }")

        # Update counters
        self.count=self.count+1 
        self.time_elapsed=self.time_elapsed+self.lat
        self.truncated = False

        # Calculate reward
        self.reward = self._calculate_reward()
        logging.info(f"LAT: {round(self.lat,4)}. RW: {self.reward}")
        if self.lat > (self.params_env["fixed_actime"] + self.tol_lat):
            logging.warning(f"WARNING: LAT is too big for current action time. Lat = {self.lat}, A_time = {self.params_env['fixed_actime']}")

        # Update episode
        if self.reward !=0:
            logging.debug(f"Episode {self.n_ep} is finished")
            self.n_ep=self.n_ep+1
        
        # Observation conversion for consistency
        self.observation = np.array(list(self.observation.values()), dtype=np.float32)

        # Add additional information (optional)
        self.info = {"terminated": self.terminated, "truncated": self.truncated, "linear_speed":self.action_dic["linear"]}

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
        self.observation, _ato = self._commstoagent.resetGetObs(timeout = 20.0)

        # Reset counters and termination flags
        self.terminated = False
        self.truncated = False 
        self.count=0
        self.time_elapsed=0
        self.crash_flag = False

        logging.info(f"Obs rec RESET: { {key: round(value, 2) for key, value in self.observation.items()} }")
        
        # Convert observation to numpy array
        self.observation = np.array(list(self.observation.values()), dtype=np.float32)

        # Add additional information (optional)
        self.info = {}

        # SB3 learn method needs a tuple
        return (self.observation, self.info) 
    

    def _compute_adjusted_reward(self, max_reward):
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
    
    
    def _calculate_reward(self):
        """
        Private method to calculate the reward based on distance.

        Args: None

        Returns:
            reward (float): The computed reward.
        """
        laser_obs = list(self.observation.values())[-4:]
        distance = self.observation["distance"]

        if self.crash_flag:
            logging.info("Crashed detected during the movement")
            self.collision_flag = True
            self.terminated=True
            return self.params_env["crash_penalty"]

        if self.params_env["finish_episode_flag"]:
            if self.action_dic["finish_flag"]<0.5:
                logging.info("Agent self truncated.")
                self.truncated = True
                if distance>self.params_env["dist_thresh_finish_flag"]:
                    return self.params_env["finish_flag_penalty"]
                else:
                    return 0

        if np.min(laser_obs)<self.params_env["max_crash_dist"]:
            logging.info("Crashed")
            self.collision_flag = True
            self.terminated=True
            return self.params_env["crash_penalty"]
        
        else:
            self.collision_flag = False

        if distance < self.params_env["reward_dist_3"]:
            self.terminated = True
            return self._compute_adjusted_reward(self.params_env["reward_3"])
        elif distance < self.params_env["reward_dist_2"]:
            self.terminated = True
            return self._compute_adjusted_reward(self.params_env["reward_2"])
        elif distance < self.params_env["reward_dist_1"]:
            self.terminated = True
            return self._compute_adjusted_reward(self.params_env["reward_1"])
        elif distance > self.params_env["max_dist"] or self.time_elapsed > self.params_env["max_time"]:
            self.terminated = True
            logging.info("Max dist or max time achieved")
            self.max_achieved = True
            return self.params_env["overlimit_penalty"]
        else:
            self.terminated = False
            self.truncated = False
            self.max_achieved = False
            self.collision_flag = False
            return 0
        

    # def _calculate_reward(self):
    #     """
    #     Private method to calculate the reward based on distance.

    #     Args: None

    #     Returns:
    #         reward (float): The computed reward.
    #     """
    #     laser_obs = list(self.observation.values())[-4:]
    #     distance = self.observation["distance"]

    #     if self.params_env["finish_episode_flag"]:
    #         if self.action_dic["finish_flag"]<0.5:
    #             logging.info("Agent self truncated.")
    #             self.truncated = True
    #             if distance>self.params_env["dist_thresh_finish_flag"]:
    #                 return self.params_env["finish_flag_penalty"]
    #             else:
    #                 return 0

    #     if np.min(laser_obs)<self.params_env["max_crash_dist"]:
    #         logging.info("Crashed")
    #         self.terminated=True
    #         return self.params_env["crash_penalty"]

    #     if distance < self.params_env["reward_dist_3"]:
    #         self.terminated = True
    #         return self.params_env["reward_3"]
    #     elif distance < self.params_env["reward_dist_2"]:
    #         self.terminated = True
    #         return self.params_env["reward_2"]
    #     elif distance < self.params_env["reward_dist_1"]:
    #         self.terminated = True
    #         return self.params_env["reward_1"]
    #     elif distance > self.params_env["max_dist"] or self.time_elapsed > self.params_env["max_time"]:
    #         self.terminated = True
    #         logging.info("Max dist or max time achieved")
    #         return self.params_env["overlimit_penalty"]
    #     else:
    #         self.terminated = False
    #         self.truncated = False
    #         return 0
        
        

# def _calculate_reward(self):
#         """
#         Private method to calculate the reward based on distance.

#         Args: None

#         Returns:
#             reward (float): The computed reward.
#         """
#         laser_obs = list(self.observation.values())[-4:]
#         distance = self.observation["distance"]

#         if self.params_env["finish_episode_flag"]:
#             if self.action_dic["finish_flag"]<0.5:
#                 logging.info("Agent self truncated.")
#                 self.truncated = True
#                 if distance>self.params_env["dist_thresh_finish_flag"]:
#                     return self.params_env["finish_flag_penalty"]
#                 else:
#                     return 0

#         if np.min(laser_obs)<self.params_env["max_crash_dist"]:
#             logging.info("Crashed")
#             self.truncated=True
#             return self.params_env["crash_penalty"]

#         if distance < self.params_env["reward_dist_3"]:
#             self.terminated = True
#             return self.params_env["reward_3"]
#         elif distance < self.params_env["reward_dist_2"]:
#             self.terminated = True
#             return self.params_env["reward_2"]
#         elif distance < self.params_env["reward_dist_1"]:
#             self.terminated = True
#             return self.params_env["reward_1"]
#         elif self.count > self.params_env["max_count"] or distance > self.params_env["max_dist"] or self.time_elapsed > self.params_env["max_time"]:
#             self.truncated = True
#             logging.info("Truncated")
#             return self.params_env["overlimit_penalty"]
#         else:
#             self.terminated = False
#             self.truncated = False
#             return 0



    def _calculate_reward_old(self):    # DEPRECATED
        """
        Private method to calculate the reward based on distance.

        Args: None

        Returns:
            reward (float): The computed reward.
        """
        distance = self.observation["distance"]

        if self.action_dic["finish_flag"]<0.5:
            logging.info("Agent decided to finish the episode.")
            self.truncated = True
            if distance>0.5:
                return -1000
            else:
                return 0
        else:
            if distance < 0.015:
                self.terminated = True
                return 10000
                
            elif distance < 0.05:
                self.terminated = True
                return 1000
                
            elif distance < 0.25:
                self.terminated = True
                return 100
                
            elif self.count>150 or distance > 2.5 or self.time_elapsed>80:
                self.truncated = True
                return -1000
                
            else:
                self.terminated = False
                self.truncated = False
                return 0
        
        
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

        # Define observation space
        if params_env["obs_time"]:
            self.observation_space= spaces.Box(low=np.array([-5,-math.pi, 0],dtype=np.float32), 
                                            high=np.array([5, math.pi, 100],dtype=np.float32), dtype=np.float32)
        else: 
            self.observation_space= spaces.Box(low=np.array([-5,-math.pi],dtype=np.float32), 
                                            high=np.array([5, math.pi],dtype=np.float32), dtype=np.float32)


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

        if params_env["obs_time"]:
            self.observation_space= spaces.Box(low=np.array([0,-math.pi, 0, 0, 0, 0, 0],dtype=np.float32), 
                                            high=np.array([5, math.pi, 4, 4, 4, 4, 100],dtype=np.float32), dtype=np.float32)
        else: 
            self.observation_space= spaces.Box(low=np.array([0,-math.pi, 0, 0, 0, 0],dtype=np.float32), 
                                            high=np.array([5, math.pi, 4, 4, 4, 4],dtype=np.float32), dtype=np.float32)