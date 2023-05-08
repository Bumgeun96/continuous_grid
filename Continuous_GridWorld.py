import numpy as np
import torch
import copy
import math
from env_physics import checking_physics

class ContinuousGridWorld:
    def __init__(self, gamma=0.99):
        # Set information about the gridworld
        self.gamma = gamma
        self.row_max = 101
        self.col_max = 101
        self.grid = np.zeros((self.row_max, self.col_max))

        # Set initial location (lower left corner state)
        self.initial_location = [self.row_max - 2, 1]
        self.agent_location = self.initial_location
        
        # Set wall
        self.wall = []
        for i in range(101):
            self.wall.append((50,i))
            self.wall.append((49,i))
            self.wall.append((51,i))
            self.wall.append((i,50))
            self.wall.append((i,49))
            self.wall.append((i,51))
        self.wall = list(set(self.wall))
        self.door = []
        for i in [x for x in range(20, 30)] + [x for x in range(71, 81)]:
            self.door.append((50,i))
            self.door.append((49,i))
            self.door.append((51,i))
            self.door.append((i,50))
            self.door.append((i,49))
            self.door.append((i,51))
        for door in self.door:
            self.wall.remove(door)
            

        # Set terminal locations : trap points & goal point (upper right corner state)
        self.trap_location1 = (2, self.col_max - 1)
        self.trap_location2 = (2, self.col_max - 2)
        self.trap_location3 = (2, self.col_max - 3)
        self.trap_location4 = (2, self.col_max - 4)
        self.trap_location5 = (1, self.col_max - 3)
        self.trap_location6 = (0, 0)
        self.trap_location7 = (1, 1)
        self.trap_location8 = (2, 2)
        self.trap_location9 = (3, 3)
        self.trap_location10 = (4, 4)
        self.trap_location11 = (4, 6)
        self.trap_location12 = (6, 4)
        self.goal_location = (55, 10)
        self.terminal_states = [self.trap_location1, self.trap_location2, self.trap_location3, self.trap_location4,
                                self.trap_location5, self.trap_location6, self.trap_location7, self.trap_location8,
                                self.trap_location9, self.trap_location10, self.trap_location11, self.trap_location12,
                                self.goal_location]

        # Set actions & initial return
        self.action_space = [-1,1]
        self.return_value = 0

    def get_reward(self,no_reward = False):        
        if tuple(np.around(self.agent_location).astype(int)) in self.terminal_states:
            reward = 100 if tuple(np.around(self.agent_location).astype(int)) == self.terminal_states[-1] else -10
            terminal = True
        else:
            reward = -1
            terminal = False
        self.return_value = reward + self.gamma * self.return_value
        if no_reward:
            reward = 0
        return reward, terminal

    def make_step(self, action, transition_prob = 0.25):
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
        self.agent_location = checking_physics(self.agent_location,previous_agent_location,self.wall)
        return self.agent_location, reward, terminal

    def reset(self):
        self.agent_location = self.initial_location
        episode_return = self.return_value
        self.return_value = 0

        return episode_return