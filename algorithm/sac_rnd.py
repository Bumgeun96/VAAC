import random
import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rl_utils.network import SoftQNetwork,Actor,rnd
from rl_utils.replay_memory import ReplayMemory as memory

import os
from gpu_scheduling import gpu_auto
idx = gpu_auto()
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= str(idx)


class sac_rnd_agent():
    def __init__(self,environment,args):
        # Initialize with args
        self.seed = args.seed
        self.buffer_size = args.buffer_size
        self.gamma = args.gamma
        self.tau = args.tau
        self.batch_size = args.batch_size
        self.learning_start = args.learning_start
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.policy_frequency = args.policy_frequency
        self.target_network_frequency = args.target_network_frequency
        self.alpha = args.alpha
        self.auto_tune = args.auto_tune
        self.latent_size = args.latent_size
        self.global_step = 0
        self.env = environment
        self.action_shape = environment.action_space.shape[0]
        self.rnd_reward_scaling = args.rnd_reward_scaling
        
        # Set a random seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        
        self.replay_memory = memory(self.buffer_size,
                                    self.batch_size,
                                    self.seed)
        
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rnd = rnd(environment,self.latent_size).to(self.device)
        self.rnd_optimizer = optim.Adam(self.rnd.model.parameters(),lr=self.critic_lr)
        self.actor = Actor(environment).to(self.device)
        self.critic1 = SoftQNetwork(environment).to(self.device)
        self.critic2 = SoftQNetwork(environment).to(self.device)
        self.critic_target1 = SoftQNetwork(environment).to(self.device)
        self.critic_target2 = SoftQNetwork(environment).to(self.device)
        self.critic_target1.load_state_dict(self.critic1.state_dict())
        self.critic_target2.load_state_dict(self.critic2.state_dict())
        self.critic_optimizer = optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=self.critic_lr)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=self.actor_lr)
        
        if self.auto_tune:
            self.target_entropy = -torch.prod(torch.Tensor(environment.action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=self.critic_lr)
        else:
            self.alpha = self.alpha

    def action(self,state):
        if self.global_step < self.learning_start:
            action = torch.rand(self.action_shape, ) * 2 - 1
        else:
            state = state.to(self.device)
            action, _, _ = self.actor.get_action(state)
        self.global_step += 1
        return action
    
    def deterministic_act(self,state):
        _, _, deterministic_action = self.actor.get_action(state)
        return deterministic_action
    
    def store_experience(self,state,action,reward,next_state,terminal,total_step):
        self.replay_memory.add(state,action,reward,next_state,terminal)

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

    def rnd_update(self,rnd_reward):
        loss = rnd_reward.mean()
        self.rnd_optimizer.zero_grad()
        loss.backward()
        self.rnd_optimizer.step()
        return loss.item()
            
    def training(self,wandb=None):
        if self.global_step > self.learning_start:
             
            # batch sampling
            experiences = self.replay_memory.sample()
            states, actions, rewards, next_states, terminations = experiences
            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            next_states = next_states.to(self.device)
            terminations = terminations.to(self.device)

            # env training
            rnd_reward = self.rnd.get_reward(states)
            rnd_loss = self.rnd_update(rnd_reward)
            rnd_reward = (rnd_reward - rnd_reward.mean())/(rnd_reward.std()+0.000001)
            if self.global_step % 1000 == 0:
                wandb.log({"rnd loss": rnd_loss},step = self.global_step)
            
            # q function training
            with torch.no_grad():
                next_actions, next_state_log_pi, _ = self.actor.get_action(next_states)
                q1_target = self.critic_target1(next_states,next_actions)
                q2_target = self.critic_target2(next_states,next_actions)
                next_Q = torch.min(q1_target,q2_target) - self.alpha*next_state_log_pi
                target = self.rnd_reward_scaling*rnd_reward + rewards+(1-terminations)*self.gamma*next_Q

            q1 = self.critic1(states,actions)
            q2 = self.critic2(states,actions)
            q1_loss = F.mse_loss(q1,target)
            q2_loss = F.mse_loss(q2,target)
            q_loss = q1_loss + q2_loss
            
            self.critic_optimizer.zero_grad()
            q_loss.backward()
            self.critic_optimizer.step()
            if self.global_step % 1000 == 0:
                wandb.log({"q loss": q_loss.item()},step=self.global_step)
            
            # TD3 delayed update
            if self.global_step % self.policy_frequency == 0:
                for _ in range(self.policy_frequency):
                    pi, log_pi, _ = self.actor.get_action(states)
                    q1_pi = self.critic1(states,pi)
                    q2_pi = self.critic2(states,pi)
                    q_pi = torch.min(q1_pi,q2_pi).view(-1)
            
                    actor_loss = (((self.alpha)*log_pi)-q_pi).mean()  
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()
                    if self.global_step % 1000 == 0:
                        wandb.log({"actor loss":actor_loss,
                                "alpha":self.alpha},
                                step=self.global_step)
                    
                    if self.auto_tune:
                        with torch.no_grad():
                            _,log_pi,_ = self.actor.get_action(states)
                        alpha_loss = (-self.log_alpha*(log_pi+self.target_entropy)).mean()
                        self.a_optimizer.zero_grad()
                        alpha_loss.backward()
                        self.a_optimizer.step()
                        self.alpha = self.log_alpha.exp().item()
                        
            if self.global_step % self.target_network_frequency == 0:
                self.soft_update(self.critic1,self.critic_target1)
                self.soft_update(self.critic2,self.critic_target2)
    
    def q_eval(self,state,action):
        state = torch.tensor(state).to(self.device)
        action = torch.tensor(action).to(self.device)
        q1 = self.critic1.q_eval(state,action)
        q2 = self.critic2.q_eval(state,action)
        q = 0.5*(q1+q2)
        return q.item()
