import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from rl_utils.network import RNDModel, PPO_ActorCritic
from rl_utils.replay_memory import RolloutBuffer

import numpy as np
from collections import defaultdict
import random

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")

class PPO:
    def __init__(self, env, args):
        self.seed = args.seed
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        lr_actor = args.actor_lr
        lr_critic = args.critic_lr
        self.gamma = args.gamma
        self.eps_clip = args.clip_coef
        self.ent_coef = args.ent_coef
        self.vf_coef = args.vf_coef
        self.max_grad_norm = args.max_grad_norm
        self.action_std_decay_freq = args.action_std_decay_freq
        self.action_std = args.action_std_init
        self.action_std_decay_rate = args.action_std_decay_rate
        self.min_action_std = args.min_action_std

        self.K_epochs = args.k_epochs
        self.episode_length = args.n_steps
        self.update_timestep = args.update_timestep
        self.time_step = 0
        
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        
        self.buffer = RolloutBuffer()
        self.visit = defaultdict(lambda: np.zeros(1))
        self.policy = PPO_ActorCritic(env, state_dim, action_dim, self.action_std).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])
        
        self.policy_old = PPO_ActorCritic(env, state_dim, action_dim, self.action_std).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        self.env = env
        
        self.rnd = RNDModel(self.env).to(device)
        self.rnd_optimizer = torch.optim.Adam(list(self.rnd.parameters()), lr=0.0001)
        
        try:
            self.env_row_max = env.row_max
            self.env_col_max = env.col_max
        except:
            pass

    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.policy_old.set_action_std(new_action_std)

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        # print("--------------------------------------------------------------------------------------------")
        self.action_std = self.action_std - action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        if (self.action_std <= min_action_std):
            self.action_std = min_action_std
            print("setting actor output action_std to min_action_std : ", self.action_std)
        else:
            print("setting actor output action_std to : ", self.action_std)
            pass
        self.set_action_std(self.action_std)


    def action(self, state):
        self.time_step += 1
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob, state_val = self.policy_old.act(state)
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)
        if self.time_step % self.action_std_decay_freq == 0:
            self.decay_action_std(self.action_std_decay_rate, self.min_action_std)

        return action.detach().flatten()

    def store_experience(self,current_state,action,reward,next_state,terminal,total_step = None):
        self.buffer.rewards.append(reward)
        self.buffer.is_terminals.append(terminal)
        intrinsic_reward = self.rnd.rnd_bonus(next_state.to(device),normalize = False)
        self.buffer.intrinsic_rewards.append(intrinsic_reward)

    def training(self):
        # Monte Carlo estimate of returns
        if self.time_step % self.update_timestep == 0:
            rewards = []
            intrinsic_rewards = []
            discounted_reward = 0
            intrinsic_discounted_reward = 0
            for reward, intrinsic_reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.intrinsic_rewards), reversed(self.buffer.is_terminals)):
                if is_terminal:
                    discounted_reward = 0
                    intrinsic_discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                intrinsic_discounted_reward = intrinsic_reward + (self.gamma * intrinsic_discounted_reward)
                rewards.insert(0, discounted_reward)
                intrinsic_rewards.insert(0, intrinsic_discounted_reward)
                
            # Normalizing the rewards
            rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
            intrinsic_rewards = torch.tensor(intrinsic_rewards, dtype=torch.float32).to(device)
            intrinsic_rewards = (intrinsic_rewards - intrinsic_rewards.mean()) / (intrinsic_rewards.std() + 1e-7)

            # convert list to tensor
            old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
            old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
            old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
            old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

            # calculate advantages
            advantages = rewards.detach() - old_state_values.detach()
            i_advantages = intrinsic_rewards.detach() - old_state_values.detach()
            advantages = advantages+i_advantages

            # Optimize policy for K epochs
            for _ in range(self.K_epochs):

                # Evaluating old actions and values
                logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

                # match state_values tensor dimensions with rewards tensor
                state_values = torch.squeeze(state_values)
                
                # Finding the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(logprobs - old_logprobs.detach())

                # Finding Surrogate Loss  
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

                # final loss of clipped objective PPO
                loss = -torch.min(surr1, surr2) + self.vf_coef* (self.MseLoss(state_values, rewards)+self.MseLoss(state_values, intrinsic_rewards)) - self.ent_coef * dist_entropy
                
                # take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
            
                self.rnd_training()
                
            # Copy new weights into old policy
            self.policy_old.load_state_dict(self.policy.state_dict())

            # clear buffer
            self.buffer.clear()
        
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
    
    def rnd_training(self):
        state = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        rnd_loss = self.rnd.rnd_bonus(state,normalize = False).sum()
        self.rnd_optimizer.zero_grad()
        rnd_loss.backward()
        self.rnd_optimizer.step()
        
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
