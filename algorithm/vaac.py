import random
import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rl_utils.network import SoftQNetwork,Actor,RNDModel,EnvNet,Virtual_Actor
from rl_utils.replay_memory import ReplayMemory as memory

# from .utils.algorithm_utils import count_visiting, get_visiting_time, count_visitation


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
        self.rnd_frequency = args.rnd_frequency
        self.rnd_reset = args.rnd_reset
        self.im_alpha = args.im_alpha
        self.beta = args.im_beta
        self.auto_tune = args.auto_tune
        self.beta_scheduling = args.beta_scheduling
        self.beta_init = args.beta_init
        self.beta_decay_freq = args.beta_decay_freq
        self.beta_decay_rate = args.beta_decay_rate
        self.min_beta = args.min_beta
        try:
            self.no_offset = args.no_offset
        except:
            self.no_offset = False
        self.global_step = 0
        self.env = environment
        self.action_shape = environment.action_space.shape[0]
        self.rnd_batch = []
        
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
            self.target_entropy = -torch.prod(torch.Tensor(environment.action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=self.critic_lr)
        else:
            self.alpha = self.alpha
        
        if self.beta_scheduling:
            self.beta = args.beta_init
        
        self.rnd = RNDModel(environment).to(self.device)
        self.env_net = EnvNet(environment).to(self.device)
        self.rnd_optimizer = optim.Adam(list(self.rnd.parameters()), lr=0.0001)
        self.env_net_optimizer = optim.Adam(list(self.env_net.parameters()), lr=0.001)
        
        self.virtual_actor = Virtual_Actor(environment).to(self.device)
        self.virtual_actor_optimizer = optim.Adam(self.virtual_actor.parameters(), lr=0.001)
            
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
        if self.rnd_reset:
            if total_step%self.rnd_frequency==0 and len(self.replay_memory)>self.batch_size:
                self.rnd.rnd_reset()
                for _ in range(self.rnd_frequency):
                    e = self.rnd_training()
        
    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
    
        
    def rnd_training(self):
        experiences = self.replay_memory.sample()
        states, actions, _, next_states, _ = experiences
        rnd_loss = self.rnd.rnd_bonus(states,normalize = False).sum()
        self.rnd_optimizer.zero_grad()
        rnd_loss.backward()
        self.rnd_optimizer.step()
        return rnd_loss.item()
    
    def rnd_training_buffer(self,state):
        rnd_loss = self.rnd.rnd_bonus(state,normalize = False).sum()
        self.rnd_optimizer.zero_grad()
        rnd_loss.backward()
        self.rnd_optimizer.step()
    
    
    def env_training(self,states,actions,next_states):
        env_loss = self.env_net.objective_function(states,actions,next_states)
        self.env_net_optimizer.zero_grad()
        env_loss.backward()
        self.env_net_optimizer.step()
        return env_loss.item()
    
    def virtual_actor_training(self,states):
        im_action, log_im_action, _ = self.virtual_actor.virtual_action(states)
        rnd_error = self.rnd.rnd_bonus(self.env_net(states,im_action),normalize=False)
        virtual_loss = (self.im_alpha*log_im_action-rnd_error).mean()
        self.virtual_actor_optimizer.zero_grad()
        virtual_loss.backward()
        self.virtual_actor_optimizer.step()
    
    def normalize(self,x):
        x = (x-x.mean())/(x.std()+0.0000001)
        return x
            
    def training(self):
        if self.global_step > self.batch_size:
            experiences = self.replay_memory.sample()
            states, actions, rewards, next_states, terminations = experiences
            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            next_states = next_states.to(self.device)
            terminations = terminations.to(self.device)
            
            #rnd training
            if not self.rnd_reset:
                self.rnd_training_buffer(states)
            
            # env prediction
            env_loss = self.env_training(states,actions,next_states)
            if self.global_step % 1000 == 0:
                print(env_loss)

            #virtual action
            if self.global_step % self.policy_frequency == 0:
                for _ in range(self.policy_frequency):
                    self.virtual_actor_training(states)
            
        if self.global_step > self.learning_start:
            with torch.no_grad():
                next_actions, next_state_log_pi, _ = self.actor.get_action(next_states)
                # boundary = self.rnd.rnd_bonus(self.env_net(next_states,next_actions),normalize=False)-self.rnd.rnd_bonus(next_states,normalize=False)
                _, im_next_state_log_pi, _ = self.virtual_actor.virtual_action(next_states)
                # im_next_state_log_pi = im_next_state_log_pi-self.rnd.rnd_bonus(next_states,normalize=False)
                q1_target = self.critic_target1(next_states,next_actions)
                q2_target = self.critic_target2(next_states,next_actions)
                # boundary = torch.clamp(boundary,min=0)
                # next_Q = torch.min(q1_target,q2_target) + self.beta*(boundary)
                if self.no_offset:
                    next_Q = torch.min(q1_target,q2_target) + self.beta*(im_next_state_log_pi)
                else:
                    next_Q = torch.min(q1_target,q2_target) + 0*self.beta*(im_next_state_log_pi-min(im_next_state_log_pi).detach())
                target = rewards+(1-terminations)*self.gamma*next_Q

            q1 = self.critic1(states,actions)
            q2 = self.critic2(states,actions)
            q1_loss = F.mse_loss(q1,target)
            q2_loss = F.mse_loss(q2,target)
            q_loss = q1_loss + q2_loss
            
            self.critic_optimizer.zero_grad()
            q_loss.backward()
            self.critic_optimizer.step()
            
            if self.beta_scheduling and self.global_step % self.beta_decay_freq == 0:
                self.beta -= self.beta_decay_rate
                self.beta = max(self.beta,self.min_beta)
            
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
    
    def get_rnd_error(self):
        visiting_table = torch.tensor(self.get_visiting_time())
        nonzero_indices = torch.nonzero(visiting_table).tolist()
        rnd_table = np.zeros((self.env_row_max, self.env_col_max))
        for row in range(self.env_row_max):
            for col in range(self.env_col_max):
                state = torch.tensor([row,col],dtype = torch.float32).cuda()
                state = self.env.normalize(state)
                rnd_error = self.rnd.rnd_bonus(state,False)
                # rnd_table[row][col] = rnd_error
                if [row,col] in nonzero_indices:
                    rnd_table[row][col] = rnd_error
                else:
                    rnd_table[row][col] = None
        return rnd_table
    
    
    def get_entropy(self):
        visiting_table = torch.tensor(self.get_visiting_time())
        nonzero_indices = torch.nonzero(visiting_table).tolist()
        entropy_table = np.zeros((self.env_row_max, self.env_col_max))
        for row in range(self.env_row_max):
            for col in range(self.env_col_max):
                state = torch.tensor([row,col],dtype = torch.float32).cuda()
                state = self.env.normalize(state)
                entropy = self.virtual_actor.get_entropy(state).detach().cpu().item()
                entropy_table[row][col] = entropy
                if [row,col] in nonzero_indices:
                    entropy_table[row][col] = entropy
                else:
                    entropy_table[row][col] = None
        return entropy_table
    
    def get_policy_entropy(self):
        visiting_table = torch.tensor(self.get_visiting_time())
        nonzero_indices = torch.nonzero(visiting_table).tolist()
        entropy_table = np.zeros((self.env_row_max, self.env_col_max))
        for row in range(self.env_row_max):
            for col in range(self.env_col_max):
                state = torch.tensor([row,col],dtype = torch.float32).cuda()
                _, log_prob, _ = self.actor.get_action(state)
                if [row,col] in nonzero_indices:
                    entropy_table[row][col] = -log_prob
                else:
                    entropy_table[row][col] = None
        return entropy_table
    
    def get_Q(self):
        visiting_table = torch.tensor(self.get_visiting_time())
        nonzero_indices = torch.nonzero(visiting_table).tolist()
        Q_table = np.zeros((self.env_row_max, self.env_col_max))
        for row in range(self.env_row_max):
            for col in range(self.env_col_max):
                state = torch.tensor([row,col],dtype = torch.float32).cuda()
                state = self.env.normalize(state)
                action, _, _ = self.actor.get_action(state)
                state = state.unsqueeze(dim=0)
                action = action.unsqueeze(dim=0)
                if [row,col] in nonzero_indices:
                    Q_table[row][col] = min(self.critic1(state,action),self.critic1(state,action))
                else:
                    Q_table[row][col] = None
        return Q_table
    
    def count_visitation(self):
        return np.count_nonzero(self.get_visiting_time())

        