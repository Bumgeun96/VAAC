import random
import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rl_utils.network import SoftQNetwork,Actor,RNDModel,EnvNet,AdventureNet
from rl_utils.replay_memory import ReplayMemory as memory


class Random_action_agent():
    def __init__(self,environment,args):
        # Initialize with args
        self.seed = args.seed
        self.env = environment
        self.action_shape = environment.action_space.shape[0]
        
        # Set a random seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        
        self.visit = defaultdict(lambda: np.zeros(1))
        try:
            self.env_row_max = environment.row_max
            self.env_col_max = environment.col_max
        except:
            pass
        
    def action(self,state):
        action = torch.rand(self.action_shape, ) * 2 - 1
        return action
    
    def store_experience(self,state,action,reward,next_state,terminal):
        pass
        
    def training(self):
        pass
                
    def count_visiting(self,state):
        self.visit[(int(np.around(state)[0]),int(np.around(state)[1]))] += 1

    def get_visiting_time(self):
        visit_table = np.zeros((self.env_row_max, self.env_col_max))
        for row in range(self.env_row_max):
            for col in range(self.env_col_max):
                if (row,col) in self.env.wall:
                    visit_table[row][col] = 0
                else:
                    visit_table[row][col] = self.visit[(row,col)]
        return visit_table
    
    def count_visitation(self):
        return np.count_nonzero(self.get_visiting_time())

        