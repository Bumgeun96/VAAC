import random
import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rl_utils.network import SoftQNetwork,Actor,Virtual_Actor,dynamics_model,re3
from rl_utils.replay_memory import ReplayMemory as memory


import os
from gpu_scheduling import gpu_auto
IDX = gpu_auto()


class VAAC_agent():
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
        self.im_alpha = args.im_alpha
        self.beta = args.im_beta
        self.auto_tune = args.auto_tune
        self.beta_scheduling = args.beta_scheduling
        self.beta_init = args.beta_init
        self.beta_decay_freq = args.beta_decay_freq
        self.beta_decay_rate = args.beta_decay_rate
        self.min_beta = args.min_beta
        self.latent_size = args.latent_size
        self.global_step = 0
        self.env = environment
        self.action_shape = environment.action_space.shape[0]
        
        # Set a random seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        
        self.replay_memory = memory(self.buffer_size,
                                    self.batch_size,
                                    self.seed)
        self.log_pi_memory = []
        
        
        self.device = torch.device('cuda:'+str(IDX) if torch.cuda.is_available() else "cpu")

        self.random_encoder = re3(self.latent_size,self.latent_size).to(self.device)

        self.dynamics = dynamics_model(environment,self.latent_size).to(self.device)
        self.dynamics_target = dynamics_model(environment,self.latent_size).to(self.device)
        self.dynamics_target.load_state_dict(self.dynamics.state_dict())
        self.dynamics_optimizer = optim.Adam(self.dynamics.parameters(),lr = 0.001,weight_decay=0.0000)

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
        
        if self.beta_scheduling:
            self.beta = args.beta_init
        
        self.virtual_actor = Virtual_Actor(environment).to(self.device)
        self.virtual_actor_optimizer = optim.Adam(self.virtual_actor.parameters(), lr=0.001)
            
    def action(self,state):
        if self.global_step < self.learning_start:
            action = torch.rand(self.action_shape, ) * 2 - 1
            # self.store_logpi(-10)
        else:
            state = state.to(self.device)
            action, log_pi, _ = self.actor.get_action(state)
            # self.store_logpi(log_pi.item())
        self.global_step += 1
        return action
    
    def store_logpi(self,log_pi):
        self.log_pi_memory.append(log_pi)
    
    def deterministic_act(self,state):
        _, _, deterministic_action = self.actor.get_action(state)
        return deterministic_action
    
    def store_experience(self,state,action,reward,next_state,terminal,total_step):
        # feature = self.random_encoder(state).detach()
        self.replay_memory.add(state,action,reward,next_state,terminal)

        
    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

    def dynamics_training(self,states,actions,next_states):
        z = self.dynamics(states,actions)
        with torch.no_grad():
            z_prime = self.dynamics_target(next_states)
        d_loss = F.mse_loss(z,z_prime)
        self.dynamics_optimizer.zero_grad()
        d_loss.backward()
        self.dynamics_optimizer.step()
        return d_loss.item()
    
    def virtual_actor_training(self,states):
        im_action, log_im_action, _ = self.virtual_actor.virtual_action(states)
        feats = self.random_encoder(self.dynamics_target(states,im_action))
        target_feats = self.random_encoder(self.dynamics_target(states))
        re_score = self.random_encoder.compute_state_entropy(feats,target_feats,K=50)
        virtual_loss = (self.im_alpha*log_im_action-re_score).mean()
        self.virtual_actor_optimizer.zero_grad()
        virtual_loss.backward()
        self.virtual_actor_optimizer.step()
        return virtual_loss.item()
    
    def novelty_diff(self,states,next_states):
        feats = self.random_encoder(self.dynamics_target(states))
        next_feats = self.random_encoder(self.dynamics_target(next_states))
        state_novelty = self.random_encoder.compute_state_entropy(feats,feats,K=50)
        next_state_novelty = self.random_encoder.compute_state_entropy(next_feats,next_feats,K=50)
        sum_novelty_diff = (next_state_novelty-state_novelty).sum().item()
        mean_novelty_diff = torch.abs(next_state_novelty-state_novelty).mean().item()
        return sum_novelty_diff, mean_novelty_diff
    
    def novelty_diff_is(self,states,next_states,idx):
        feats = self.random_encoder(self.dynamics_target(states))
        next_feats = self.random_encoder(self.dynamics_target(next_states))
        state_novelty = self.random_encoder.compute_state_entropy(feats,feats,K=50)
        next_state_novelty = self.random_encoder.compute_state_entropy(next_feats,next_feats,K=50)
        log_pi = [self.log_pi_memory[i] for i in idx]
        log_pi = torch.tensor(log_pi).to(self.device)
        pi = torch.exp(log_pi)
        avg_nov = ((next_state_novelty-state_novelty)/pi).mean().item()*0.001
        return avg_nov
    
    def diff_err(self,states):
        im_action,log_im_action,_=self.virtual_actor.virtual_action(states)
        next_feats = self.random_encoder(self.dynamics_target(states,im_action))
        feats = self.random_encoder(self.dynamics_target(states))
        state_novelty = self.random_encoder.compute_state_entropy(feats,feats,K=50)
        next_state_novelty = self.random_encoder.compute_state_entropy(next_feats,next_feats,K=50)
        diff = (next_state_novelty-state_novelty)
        (diff-diff.mean())/(diff.std())

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
            env_loss = self.dynamics_training(states,actions,next_states)
            if self.global_step % 1000 == 0:
                wandb.log({"env loss": env_loss},step = self.global_step)
            
            # q function training
            with torch.no_grad():
                next_actions, next_state_log_pi, _ = self.actor.get_action(next_states)
                _, im_next_state_log_pi, _ = self.virtual_actor.virtual_action(next_states)
                q1_target = self.critic_target1(next_states,next_actions)
                q2_target = self.critic_target2(next_states,next_actions)
                next_Q = torch.min(q1_target,q2_target) - self.alpha*next_state_log_pi + self.beta*(im_next_state_log_pi-((im_next_state_log_pi).mean()/(im_next_state_log_pi).std()).detach())
                target = rewards+(1-terminations)*self.gamma*next_Q

            q1 = self.critic1(states,actions)
            q2 = self.critic2(states,actions)
            q1_loss = F.mse_loss(q1,target)
            q2_loss = F.mse_loss(q2,target)
            q_loss = q1_loss + q2_loss
            
            self.critic_optimizer.zero_grad()
            q_loss.backward()
            self.critic_optimizer.step()
            if self.global_step % 1000 == 0:
                # avg_novelty_diff = self.novelty_diff_is(states,next_states,idx)
                sum_novelty_diff, novelty_diff = self.novelty_diff(states,next_states)
                wandb.log({"novelty diff": novelty_diff},step=self.global_step)
                # wandb.log({"average novelty diff": avg_novelty_diff},step=self.global_step)
                wandb.log({"sum novelty diff": sum_novelty_diff},step=self.global_step)
                wandb.log({"q loss": q_loss.item()},step=self.global_step)

            
            if self.beta_scheduling and self.global_step % self.beta_decay_freq == 0:
                self.beta -= self.beta_decay_rate
                self.beta = max(self.beta,self.min_beta)
            
            # TD3 delayed update
            if self.global_step % self.policy_frequency == 0:
                for _ in range(self.policy_frequency):
                    virtual_loss = self.virtual_actor_training(states)
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
                                "virtual actor loss":virtual_loss,
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
                self.soft_update(self.dynamics,self.dynamics_target)
    
    def q_eval(self,state,action):
        state = torch.tensor(state).to(self.device)
        action = torch.tensor(action).to(self.device)
        q1 = self.critic1.q_eval(state,action)
        q2 = self.critic2.q_eval(state,action)
        q = 0.5*(q1+q2)
        return q.item()
    