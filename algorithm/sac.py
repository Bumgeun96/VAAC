# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import random
import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rl_utils.network import SoftQNetwork,Actor,discriminator
from rl_utils.replay_memory import ReplayMemory as memory


class SAC_agent():
    def __init__(self,environment,args):
        # Initialize with args
        self.algo = args.algo
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
        self.noise_clip = args.noise_clip
        self.alpha = args.alpha
        self.auto_tune = args.auto_tune
        self.mda_alpha = args.mda_alpha
        self.cliping_discriminator = args.cliping_discriminator
        self.d_step = args.d_step
        self.global_step = 0
        self.action_shape = environment.action_space.shape[0]
        
        # Set a random seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        
        self.replay_memory = memory(self.buffer_size,self.batch_size,self.seed)
        self.visit = defaultdict(lambda: np.zeros(1))
        try:
            self.env_row_max = environment.row_max
            self.env_col_max = environment.col_max
        except:
            pass
        
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
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
            self.target_entropy = -torch.prod(torch.Tensor(self.action_shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=self.critic_lr)
        else:
            self.alpha = self.alpha
        
        if self.algo:
            self.discriminator = discriminator(environment,epsilon=self.cliping_discriminator).to(self.device)
            self.discriminator_optimizer = optim.Adam(list(self.discriminator.parameters()), lr=0.001)
            
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
    
    def store_experience(self,state,action,reward,next_state,terminal):
        self.replay_memory.add(state,action,reward,next_state,terminal)
        
    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
        
    def training(self):
        if self.global_step > self.learning_start:
            experiences = self.replay_memory.sample()
            states, actions, rewards, next_states, terminations = experiences
            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            next_states = next_states.to(self.device)
            terminations = terminations.to(self.device)
            
            if self.algo:
                if self.global_step % self.d_step == 0:
                    for _ in range(self.d_step):
                        d_m = self.discriminator(states,actions)
                        with torch.no_grad():
                            p_act, _, _ = self.actor.get_action(states)
                        d_p = self.discriminator(states,p_act)
                        discriminate_loss = -(torch.log(1-d_m)+torch.log(d_p)).mean()
                        self.discriminator_optimizer.zero_grad()
                        discriminate_loss.backward()
                        self.discriminator_optimizer.step()
                    
            
            with torch.no_grad():
                next_actions, next_state_log_pi, _ = self.actor.get_action(next_states)
                q1_target = self.critic_target1(next_states,next_actions)
                q2_target = self.critic_target2(next_states,next_actions)
                if self.algo:
                    next_Q = torch.min(q1_target,q2_target)+self.mda_alpha*torch.log(self.discriminator(next_states,next_actions))
                    target = rewards+self.mda_alpha*torch.log(self.discriminator(states,actions))+(1-terminations)*self.gamma*next_Q
                else:
                    next_Q = torch.min(q1_target,q2_target) - self.alpha*next_state_log_pi
                    target = rewards+(1-terminations)*self.gamma*next_Q

            q1 = self.critic1(states,actions)
            q2 = self.critic2(states,actions)
            q1_loss = F.mse_loss(q1,target)
            q2_loss = F.mse_loss(q2,target)
            q_loss = q1_loss + q2_loss
            
            self.critic_optimizer.zero_grad()
            q_loss.backward()
            self.critic_optimizer.step()
            
            # TD3 delayed update
            if self.global_step % self.policy_frequency == 0:
                for _ in range(self.policy_frequency):
                    pi, log_pi, _ = self.actor.get_action(states)
                    q1_pi = self.critic1(states,pi)
                    q2_pi = self.critic2(states,pi)
                    q_pi = torch.min(q1_pi,q2_pi).view(-1)
                    if self.algo:
                        score_prop = torch.log(self.discriminator(states,pi))
                        actor_loss = ((self.alpha*log_pi)-q_pi-self.mda_alpha*score_prop).mean()
                    else:
                        actor_loss = ((self.alpha*log_pi)-q_pi).mean()
                    
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()
                    
                    if self.auto_tune:
                        with torch.no_grad():
                            _,log_pi,_ = self.actor.get_action(states)
                        alpha_loss = (-self.log_alpha*(log_pi+self.target_entropy)).mean()
                        
                        self.a_optimizer.zero_grad()
                        alpha_loss.backward()
                        self.a_optimizer.step()
                        self.alpha = self.log_alpha.exp().item()
            # if self.algo:
            #     try:
            #         condition = torch.exp(score_prop)>0.8
            #         try:
            #             idx = torch.nonzero(condition).squeeze()[:,0]
            #         except:
            #             idx = torch.nonzero(condition).squeeze()[0]
            #         # print(states[idx])
            #         print(pi[idx])
            #     except:
            #         pass
            
            if self.global_step % self.target_network_frequency == 0:
                self.soft_update(self.critic1,self.critic_target1)
                self.soft_update(self.critic2,self.critic_target2)
                
    def count_visiting(self,state):
        self.visit[(int(np.around(state)[0]),int(np.around(state)[1]))] += 1

    
    def get_visiting_time(self):
        visit_table = np.zeros((self.env_row_max, self.env_col_max))
        for row in range(self.env_row_max):
            for col in range(self.env_col_max):
                visit_table[row][col] = self.visit[(row,col)]
        return visit_table
    
    def count_visitation(self):
        return np.count_nonzero(self.get_visiting_time())