from collections import namedtuple, deque
import random
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayMemory:
    def __init__(self, memory_size, batch_size, seed=0):
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "termination"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, termination):
        experience = self.experience(state, action, reward, next_state, termination)
        self.memory.append(experience)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        # experiences_idx = random.sample(range(len(self.memory)), k=self.batch_size)
        # experiences = [self.memory[i] for i in experiences_idx]

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        terminations = torch.from_numpy(np.vstack([e.termination for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, terminations)
        # return (states, actions, rewards, next_states, terminations),experiences_idx
    
    def __len__(self):
        return len(self.memory)
    
class ReplayMemory_feature:
    def __init__(self, memory_size, batch_size, state_shape, action_shape, latent_size, seed=0):
        self.state = np.empty((memory_size,state_shape))
        self.action = np.empty((memory_size,action_shape))
        self.reward = np.empty((memory_size,1))
        self.next_state = np.empty((memory_size,state_shape))
        self.termination = np.empty((memory_size,1))
        self.feature = np.empty((memory_size,latent_size))
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.seed = random.seed(seed)
        self.idx = 0
        self.full = False
        self.K = 50

    def add(self, state, action, reward, next_state, termination, feature):
        self.state[self.idx] = state
        self.action[self.idx] = action
        self.reward[self.idx] = reward
        self.next_state[self.idx] = next_state
        self.termination[self.idx] = termination
        self.feature[self.idx] = feature
        if self.idx + 1 == self.memory_size:
            self.full = True
        self.idx = (self.idx + 1) % self.memory_size

    def compute_state_entropy(self,src_feats,tgt_feats,K=None):
        if K == None:
            k = self.k
        else:
            k = K
        with torch.no_grad():
            dists = []
            for idx in range(len(tgt_feats)//10000+1):
                start = idx*10000
                end = (idx+1)*10000
                dist = torch.norm(
                    src_feats[:,None,:]-tgt_feats[None,start:end,:],
                    dim=-1,
                    p=2)
                dists.append(dist)
            dists = torch.cat(dists,dim=1)
            knn_dists = 0.0
            for k in range(k):
                knn_dists += torch.kthvalue(dists,k+1,dim=1).values
            knn_dists /= k
            state_entropy = knn_dists
        return state_entropy.unsqueeze(1)

    def sample(self):
        if self.full:
            idxs = np.random.randint(0,self.memory_size-1,size=self.batch_size)
        else:
            idxs = np.random.randint(0,self.idx,size=self.batch_size)
        states = torch.tensor(self.state[idxs]).float()
        actions = torch.tensor(self.action[idxs]).float()
        rewards = torch.tensor(self.reward[idxs]).float()
        next_states = torch.tensor(self.next_state[idxs]).float()
        terminations = torch.tensor(self.termination[idxs]).float()
        features = torch.tensor(self.feature[idxs]).float().to(device)
        # if self.full:
        #     target_features = torch.tensor(self.feature).float().to(device)
        # else:
        #     target_features = torch.tensor(self.feature[:self.idx]).float().to(device)
        re_score = self.compute_state_entropy(features,features,K=50)
        return (states, actions, rewards, next_states, terminations, re_score)
    
    def __len__(self):
        return len(self.memory)
    
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
        self.intrinsic_rewards = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        del self.intrinsic_rewards[:]