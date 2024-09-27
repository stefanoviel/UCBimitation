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
class DiscreteGaussianGridWorld(TabularEnv):
    def __init__(self, env_type=0, prop=0):
        # Characteristics of the gridworld

        TabularEnv.__init__(self, prop)
        self.env_type = env_type
        # env_type 0: Random start within grid, terminal area in lower-left quadrant, reward focuses on reaching terminal area and minimizing distance from origin, large reward when terminal area is reached.
        # env_type 1: Fixed start at (-1, 1), terminal area in upper-right quadrant, reward focuses on minimizing distance from (1, -1), additional reward for reaching terminal area.
        
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
        #print(a, "action")
        reward = self.compute_reward()
        if self.done:
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
                        ]
                        ), \
               reward, \
               self.done, \
               {}
        
        self.state += self._action_to_direction[a] + self.prop*np.random.uniform(-0.1, 0.1, size=2)
        self.state[0] = np.max([np.min([1,self.state[0]]),-1])
        self.state[1] = np.max([np.min([1, self.state[1]]), -1])
        if (self.terminal_area[0,0] <= self.state[0]  <= self.terminal_area[0,1] and
            self.terminal_area[1,0] <= self.state[1]  <= self.terminal_area[1,1]):
            self.done = True
        
        self.steps_from_last_reset += 1
        """ if self.steps_from_last_reset == 5000:
            self.done = True
            return np.array([self.state[0],
                             self.state[1],
                             # self.state[0]*self.state[1],
                             self.state[0] ** 2,
                             self.state[1] ** 2,
                             # self.state[0] ** 3,
                             # self.state[1] ** 3,
                             8*np.exp(-8*self.state[0]**2-8*self.state[1]**2),
                             0.0]
                            ), \
                   reward, \
                   self.done, \
                   None """
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
                        ]
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
                        self.state[0]*self.state[1],
                        self.state[0]**2,
                        self.state[1]**2,
                        1,
                        #self.state[0] ** 3,
                        #self.state[1] ** 3,
                        8*np.exp(-8*self.state[0]**2-8*self.state[1]**2),
                        float(self.done)
                        ]
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
            reward = -(self.state[0] -1)**2 - (self.state[1] + 1)**2 - 80*np.exp(-8*self.state[0]**2-8*self.state[1]**2)
            if (self.terminal_area[0, 0] <= self.state[0] <= self.terminal_area[
                0, 1] and
                    self.terminal_area[1, 0] <= self.state[1] <=
                    self.terminal_area[1, 1]):
                reward += 100
        return reward