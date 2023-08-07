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
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        terminations = torch.from_numpy(np.vstack([e.termination for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, terminations)

    def __len__(self):
        return len(self.memory)