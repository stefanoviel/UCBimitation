import math
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import copy
import random
from scipy import sparse

# TODO: which environemnt should I use

class TabularEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }


    def __init__(self, prop):

        self.viewer = None

        self.reward_range = (-100, 0)
        # although there are 2 terminal squares in the grid
        # they are considered as 1 state
        # therefore observation is between 0 and 14
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([0.01, 0]),
            1: np.array([0, 0.01]),
            2: np.array([-0.01, 0]),
            3: np.array([0, -0.01]),
            # 4: np.array([0.02, 0]),
            # 5: np.array([0, 0.02]),
            # 6: np.array([-0.02, 0]),
            # 7: np.array([0, -0.02]),
            # 8: np.array([0.03, 0]),
            # 9: np.array([0, 0.03]),
            # 10: np.array([-0.03, 0]),
            # 11: np.array([0, -0.03]),
            # 12: np.array([0.05, 0]),
            # 13: np.array([0, 0.05]),
            # 14: np.array([-0.05, 0]),
            # 15: np.array([0, -0.05]),
            # 16: np.array([0.18, 0]),
            # 17: np.array([0, 0.18]),
            # 18: np.array([-0.18, 0]),
            # 19: np.array([0, -0.18]),
        }
        self.observation_space = spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)

        # Discount factor
        self.gamma = 0.99

        # Stochastic transitions
        self.prop_random_actions = prop

class DiscreteContinuousGridWorld(TabularEnv):
    def __init__(self, env_type=0, prop=0):
        # Characteristics of the gridworld

        TabularEnv.__init__(self, prop)
        self.env_type = env_type
        self.state=None
        self.prop=prop
        self.env_type=env_type
        self.seed()
        self.reset()
        self.steps_from_last_reset=0
        if env_type == 0:
            self.terminal_area = np.array([[-1.0, -0.95], [0.95, 1.0]])
        else:
            self.terminal_area = np.array([[0.95, 1.0],[-1.0, -0.95]])

    def step(self, a):
        self.state += self._action_to_direction[a] + self.prop*np.random.uniform(-0.1, 0.1, size=2)
        self.state[0] = np.max([np.min([1,self.state[0]]),-1])
        self.state[1] = np.max([np.min([1, self.state[1]]), -1])
        if (self.terminal_area[0,0] <= self.state[0]  <= self.terminal_area[0,1] and
            self.terminal_area[1,0] <= self.state[1]  <= self.terminal_area[1,1]):
            self.done = True
        reward = self.compute_reward()
        self.steps_from_last_reset += 1
        if self.steps_from_last_reset == 5000:
            self.done = True
            return np.array([self.state[0],
                             self.state[1],
                             # self.state[0]*self.state[1],
                             self.state[0] ** 2,
                             self.state[1] ** 2,
                             # self.state[0] ** 3,
                             # self.state[1] ** 3,
                             (1 / (self.state[0] ** 2
                                            + self.state[1] ** 2
                                            + 1e-8)) ** 2,
                             0.0
                             ]
                            ), \
                   reward, \
                   self.done, \
                   {}
        return np.array([self.state[0],
                        self.state[1],
                        #self.state[0]*self.state[1],
                        self.state[0]**2,
                        self.state[1]**2,
                        #self.state[0] ** 3,
                        #self.state[1] ** 3,
                        (1 / (self.state[0] ** 2
                               + self.state[1] ** 2
                               + 1e-8)) ** 2,
                        10.*float(self.done)]
                        ), \
               reward, \
               self.done, \
               {}

    def reset(self, starting_index = None):
        if self.env_type == 0:
            self.state = np.random.uniform(-1, 1, size=2)
        else:
            self.state = np.array([-1.0, 1.0])
        self.steps_from_last_reset = 0
        self.done = False
        return np.array([self.state[0],
                        self.state[1],
                        #self.state[0]*self.state[1],
                        self.state[0]**2,
                        self.state[1]**2,
                        #self.state[0] ** 3,
                        #self.state[1] ** 3,
                        (1 / (self.state[0] ** 2
                               + self.state[1] ** 2
                               + 1e-8)) ** 2,
                        100.*float(self.done)]
                        )
    def compute_reward(self):
        if self.env_type==0:
            if (self.terminal_area[0,0] <= self.state[0]  <= self.terminal_area[0,1] and
            self.terminal_area[1,0] <= self.state[1]  <= self.terminal_area[1,1]):
                reward = -(self.state[0] ** 2 + self.state[1] ** 2) + 3 * \
                     self.state[0] - 5 + 2000
            else:
                reward = -(self.state[0] ** 2 + self.state[1] ** 2) + 3 * \
                     self.state[0] - 5
        elif self.env_type==1:
            reward = -(self.state[0] -1)**2 - (self.state[1] + 1)**2 - (1/(self.state[0]**2
                                                                 + self.state[1]**2
                                                                 + 1e-8))**2
            if (self.terminal_area[0, 0] <= self.state[0] <= self.terminal_area[
                0, 1] and
                    self.terminal_area[1, 0] <= self.state[1] <=
                    self.terminal_area[1, 1]):
                reward += 100
        return reward


# use this one
class DiscreteGaussianGridWorld(gym.Env):
    def __init__(self, env_type=0, prop=0):
        super().__init__()

        self.env_type = env_type
        self.prop = prop
        self.state = None
        self.steps_from_last_reset = 0

        # Define action and observation spaces
        self.action_space = spaces.Discrete(4)  # 4 discrete actions
        self.observation_space = spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)

        # Define action to direction mapping
        self._action_to_direction = {
            0: np.array([0, 1]),   # Up
            1: np.array([1, 0]),   # Right
            2: np.array([0, -1]),  # Down
            3: np.array([-1, 0])   # Left
        }

        if env_type == 0:
            self.terminal_area = np.array([[-1.0, -0.95], [0.95, 1.0]])
        else:
            self.terminal_area = np.array([[0.95, 1.0], [-1.0, -0.95]])


    def seed(self, seed=None):
        """Set the seed for this env's random number generator(s)."""
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def get_observation(self):
        return np.array([self.state[0],
                        self.state[1],
                        self.state[0]*self.state[1],
                        self.state[0]**2,
                        self.state[1]**2,
                        1,
                        #self.state[0] ** 3,
                        #self.state[1] ** 3,
                        8*np.exp(-8*self.state[0]**2-8*self.state[1]**2),
                        float(self.done)
                        ])
    def step(self, a):
        #print(a, "action")
        reward = self.compute_reward()
        if self.done:
            return self.get_observation(), reward, self.done, {}
        
        self.state += self._action_to_direction[a] + self.prop*np.random.uniform(-0.1, 0.1, size=2)
        self.state[0] = np.max([np.min([1,self.state[0]]),-1])
        self.state[1] = np.max([np.min([1, self.state[1]]), -1])
        if (self.terminal_area[0,0] <= self.state[0]  <= self.terminal_area[0,1] and
            self.terminal_area[1,0] <= self.state[1]  <= self.terminal_area[1,1]):
            self.done = True
        
        self.steps_from_last_reset += 1
        return self.get_observation(), reward, self.done, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.env_type == 0:
            self.state = self.np_random.uniform(-1, 1, size=2)
        else:
            self.state = np.array([-1.0, 1.0])

        self.steps_from_last_reset = 0
        self.done = False

        return self.get_observation(), {}
    

    # TODO: compute cost and update everything else
    def compute_reward(self):
        """
        Computes the reward based on the agent's position (self.state) in the environment.
        The reward function varies depending on the environment type (env_type) and whether
        the agent is within a specified terminal area (self.terminal_area).

        env_type 0:
        -----------
        - The environment rewards the agent based on its position (x, y).
        - If the agent is within the terminal area (a rectangular region defined by 
        self.terminal_area), a bonus of +2000 is added.
        - The reward is calculated as:
            reward = -(x^2 + y^2) + 3x - 5 + 2000 (if inside the terminal area)
            reward = -(x^2 + y^2) + 3x - 5 (if outside the terminal area)
        Where:
        - (x, y) is the agent's current position (self.state[0], self.state[1]).
        - The term -(x^2 + y^2) penalizes the agent for being far from the origin (0, 0).
        - The term +3x provides a linear incentive based on the x-coordinate.
        - The constant -5 is a fixed offset.
        - The bonus +2000 is awarded if the agent reaches the terminal area.

        env_type 1:
        -----------
        - The reward function is more complex and involves two components:
        1. A penalty based on the distance from the point (1, -1), calculated as:
            -(x - 1)^2 - (y + 1)^2
        2. A Gaussian-shaped negative penalty centered at the origin (0, 0), given by:
            -80 * exp(-8 * x^2 - 8 * y^2), which penalizes the agent for moving away from the origin.
        - If the agent is within the terminal area, a bonus of +100 is added.
        - The reward is calculated as:
            reward = -(x - 1)^2 - (y + 1)^2 - 80 * exp(-8 * x^2 - 8 * y^2)
            If inside the terminal area, an additional +100 is awarded.
        
        Returns:
        --------
        - The computed reward for the agent's current state in the environment.
        """

        if self.env_type==0:
            if (self.terminal_area[0,0] <= self.state[0]  <= self.terminal_area[0,1] and
            self.terminal_area[1,0] <= self.state[1]  <= self.terminal_area[1,1]):
                reward = -(self.state[0] ** 2 + self.state[1] ** 2) + 3 * \
                     self.state[0] - 5 + 2000
            else:
                reward = -(self.state[0] ** 2 + self.state[1] ** 2) + 3 * \
                     self.state[0] - 5
        elif self.env_type==1:
            reward = -(self.state[0] -1)**2 - (self.state[1] + 1)**2 - 80*np.exp(-8*self.state[0]**2-8*self.state[1]**2)
            if (self.terminal_area[0, 0] <= self.state[0] <= self.terminal_area[
                0, 1] and
                    self.terminal_area[1, 0] <= self.state[1] <=
                    self.terminal_area[1, 1]):
                reward += 100
        return reward