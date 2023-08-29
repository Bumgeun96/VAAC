import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from rl_utils.network import RNDModel

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


################################## PPO Policy ##################################
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


class ActorCritic(nn.Module):
    def __init__(self, env, state_dim, action_dim, action_std_init):
        super(ActorCritic, self).__init__()
        self.action_dim = action_dim
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        # actor
        self.actor = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, action_dim),
                        nn.Tanh()
                    )
        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )
    
        # action rescaling
        self.register_buffer(
                "action_scale", torch.tensor((env.action_space.high[0] - env.action_space.low[0]) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high[0] + env.action_space.low[0]) / 2.0, dtype=torch.float32)
        )
    
    def set_action_std(self, new_action_std):
        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)

    def forward(self):
        raise NotImplementedError
    
    def act(self, state):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)
        x = dist.sample()
        action = torch.tanh(x)* self.action_scale + self.action_bias
        action_logprob = dist.log_prob(action)
        action_logprob -= torch.log(1-action.pow(2)+1e-6).sum(dim=1)
        state_val = self.critic(state)
        return action[0].detach(), action_logprob.detach(), state_val.detach()
    
    def evaluate(self, state, action):
        action_mean = self.actor(state)
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)
        dist = MultivariateNormal(action_mean, cov_mat)
        
        # For Single Action Environments.
        if self.action_dim == 1:
            action = action.reshape(-1, self.action_dim)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, env, args):
        self.seed = args.seed
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        lr_actor = args.actor_lr
        lr_critic = args.critic_lr
        self.gamma = args.gamma
        self.action_std_decay_freq = args.action_std_decay_freq
        self.action_std = args.action_std_init
        self.action_std_decay_rate = args.action_std_decay_rate
        self.min_action_std = args.min_action_std

        self.eps_clip = args.noise_clip
        self.K_epochs = args.k_epochs
        self.episode_length = args.n_steps
        self.update_timestep = self.episode_length*4
        self.time_step = 0
        
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        
        self.buffer = RolloutBuffer()
        self.visit = defaultdict(lambda: np.zeros(1))
        self.policy = ActorCritic(env, state_dim, action_dim, self.action_std).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])
        
        self.policy_old = ActorCritic(env, state_dim, action_dim, self.action_std).to(device)
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
            # print("setting actor output action_std to min_action_std : ", self.action_std)
        else:
            pass
            # print("setting actor output action_std to : ", self.action_std)
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

    def store_experience(self,current_state,action,reward,next_state,terminal):
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
            advantages += i_advantages

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
                loss = -torch.min(surr1, surr2) + 0.5 * (self.MseLoss(state_values, rewards)+self.MseLoss(state_values, intrinsic_rewards)) - 0.01 * dist_entropy
                
                # take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
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
