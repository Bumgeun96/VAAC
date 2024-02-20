import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from rl_utils.network import RNDModel, PPO_AC
from rl_utils.replay_memory import RolloutBuffer


import numpy as np
from collections import defaultdict
import random
class ppo:
    def __init__(self,env,args):
        self.env = env
        self.seed = args.seed
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        lr_actor = args.ppo_lr
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self.batch_size = args.batch_size
        self.minibatch_size = args.minibatch_size
        self.max_grad_norm = args.max_grad_norm
        self.clip_coef = args.clip_coef
        self.ent_coef = args.ent_coef
        self.vf_coef = args.vf_coef
        self.clip_vloss = args.clip_vloss
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
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PPO_AC(env).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr_actor, eps=1e-5)
        
        self.buffer = RolloutBuffer()
        self.visit = defaultdict(lambda: np.zeros(1))
        
    def action(self,state):
        self.time_step += 1
        action, log_prob, _, value = self.policy.get_action_and_value(state.to(self.device))
        self.buffer.values.append(value.flatten())
        self.buffer.logprobs.append(log_prob)
        return action
    
    def store_experience(self,current_state,action,reward,next_state,terminal,total = None):
        self.buffer.states.append(current_state)
        self.buffer.actions.append(torch.tensor(action,dtype=torch.float32))
        self.buffer.is_terminals.append(torch.tensor(terminal,dtype=torch.float32))
        self.buffer.rewards.append(torch.tensor(reward,dtype=torch.float32))
        self.next_state = torch.tensor(next_state,dtype=torch.float32).to(self.device)
        self.done = torch.tensor(terminal,dtype=torch.float32).to(self.device)
    
    def training(self):
        if self.time_step % self.update_timestep == 0:
            obs = torch.stack(self.buffer.states,dim=0).to(self.device)
            actions = torch.stack(self.buffer.actions,dim=0).to(self.device)
            logprobs = torch.unsqueeze(torch.stack(self.buffer.logprobs,dim=0),dim=1).to(self.device)
            values = torch.stack(self.buffer.values,dim=0).to(self.device)
            rewards = torch.unsqueeze(torch.stack(self.buffer.rewards,dim=0),dim=1).to(self.device)
            dones = torch.unsqueeze(torch.stack(self.buffer.is_terminals,dim=0),dim=1).to(self.device)
            with torch.no_grad():
                next_value = self.policy.get_value(self.next_state).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(self.device)
                lastgaelam = 0
                for t, reward, value, done in zip(range(self.update_timestep),reversed(rewards),reversed(values),reversed(dones)):
                    if t == self.update_timestep-1:
                        nextnonterminal = 1.0 - self.done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t+1]
                        nextvalues = values[t+1]
                    delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            # flatten the batch
            with torch.no_grad():
                b_obs = obs.reshape((-1,) + self.env.observation_space.shape)
                b_logprobs = logprobs.reshape(-1)
                b_actions = actions.reshape((-1,) + self.env.action_space.shape)
                b_advantages = advantages.reshape(-1)
                b_returns = returns.reshape(-1)
                b_values = values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(self.batch_size)
            clipfracs = []
            for epoch in range(self.K_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_inds = b_inds[start:end]
                    _, newlogprob, entropy, newvalue = self.policy.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()
                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.clip_coef,
                            self.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.optimizer.step()
            self.buffer.clear()
