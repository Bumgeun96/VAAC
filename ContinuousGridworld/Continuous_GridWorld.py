import numpy as np
import torch
import copy
import math
import gymnasium as gym
from .env_physics import checking_physics
from .wall_setting import map_1, map_2

class ContinuousGridWorld:
    def __init__(self, gamma=0.99, map = 1):
        # Set information about the gridworld
        self.n_steps = 0
        self.gamma = gamma
        self.row_max = 101
        self.col_max = 101
        self.grid = np.zeros((self.row_max, self.col_max))

        # Set initial location (lower left corner state)
        self.initial_location = [self.row_max - 2, 1]
        self.agent_location = self.initial_location
        self.action_range = [-1,1]
        self.observation_space = gym.spaces.Box(low=0,
                                                high=self.row_max,
                                                shape=(2,))
        self.action_space = gym.spaces.Box(low=self.action_range[0],
                                           high=self.action_range[1],
                                           shape=(2,))
        
        # Set wall
        if map == 1:
            self.wall, self.boundary_points = map_1()
        elif map == 2:
            self.wall, self.boundary_points = map_2()


    def get_reward(self,no_reward = False):        

        reward = -1
        terminal = False
        if no_reward:
            reward = 0
        if self.n_steps >= 1000:
            terminal = True
        return reward, terminal

    def make_step(self, action, transition_prob = 0.25):
        self.n_steps += 1
        # introduce stochastic transition
        if np.random.uniform(0, 1) < transition_prob:
            action = np.array([np.random.uniform(-1, 1),np.random.uniform(-1, 1)])
        
        vertical_movement = action[0]
        horizontal_movement = action[1]
        previous_agent_location = copy.deepcopy(self.agent_location)
        self.agent_location = [self.agent_location[0] - vertical_movement,
                               self.agent_location[1] + horizontal_movement]

        
        if self.agent_location[0] < 0:
            x_new = 0
            y_new = self.agent_location[1]+(x_new-self.agent_location[0])\
                *(previous_agent_location[1]-self.agent_location[1])\
                    /(previous_agent_location[0]-self.agent_location[0])
            self.agent_location = [x_new,y_new]
            
        if self.agent_location[0] > self.row_max - 1:
            x_new = self.row_max - 1
            y_new = self.agent_location[1]+(x_new-self.agent_location[0])\
                *(previous_agent_location[1]-self.agent_location[1])\
                    /(previous_agent_location[0]-self.agent_location[0])
            self.agent_location = [x_new,y_new]
            
        if self.agent_location[1] < 0:
            y_new = 0
            x_new = self.agent_location[0]+(y_new-self.agent_location[1])\
                *(previous_agent_location[0]-self.agent_location[0])\
                    /(previous_agent_location[1]-self.agent_location[1])
            self.agent_location = [x_new,y_new]
            
        if self.agent_location[1] > self.col_max -1:
            y_new = self.col_max -1
            x_new = self.agent_location[0]+(y_new-self.agent_location[1])\
                *(previous_agent_location[0]-self.agent_location[0])\
                    /(previous_agent_location[1]-self.agent_location[1])
            self.agent_location = [x_new,y_new]
        
        reward, terminal = self.get_reward(no_reward=True)
        self.agent_location = checking_physics(self.agent_location,previous_agent_location,self.boundary_points)
        return self.agent_location, reward, terminal

    def reset(self):
        self.agent_location = self.initial_location
        self.n_steps = 0