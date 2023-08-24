import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class discriminator(nn.Module):
    def __init__(self, env, epsilon = 0.0001):
        super().__init__()
        self.epsilon = epsilon
        self.obs_high = torch.tensor([env.observation_space.high])
        self.obs_low = torch.tensor([env.observation_space.low])
        self.fc1 = nn.Linear(env.observation_space.shape[0]+env.action_space.shape[0], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = (x-self.obs_low.to('cuda'))/(self.obs_high-self.obs_low).to('cuda')
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.clamp(torch.sigmoid(self.fc3(x)),min=self.epsilon,max=1-self.epsilon)

class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(env.observation_space.shape[0]+env.action_space.shape[0], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(env.observation_space.shape[0], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, env.action_space.shape[0])
        self.fc_logstd = nn.Linear(256, env.action_space.shape[0])
        
        # uniform initialize
        nn.init.uniform_(self.fc1.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc1.bias, -0.1, 0.1)
        nn.init.uniform_(self.fc2.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc2.bias, -0.1, 0.1)
        nn.init.uniform_(self.fc_mean.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc_mean.bias, -0.1, 0.1)
        nn.init.uniform_(self.fc_logstd.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc_logstd.bias, -0.1, 0.1)
        
        # action rescaling
        self.register_buffer(
                "action_scale", torch.tensor((env.action_space.high[0] - env.action_space.low[0]) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high[0] + env.action_space.low[0]) / 2.0, dtype=torch.float32)
        )
            

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
    
class EnvNet(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.fc1 = nn.Linear(env.observation_space.shape[0]+env.action_space.shape[0], 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, env.observation_space.shape[0])
        self.std = nn.Linear(256, env.observation_space.shape[0])
        
        nn.init.uniform_(self.fc1.weight, -0.01, 0.01)
        nn.init.uniform_(self.fc1.bias, -0.01, 0.01)
        nn.init.uniform_(self.fc2.weight, -0.01, 0.01)
        nn.init.uniform_(self.fc2.bias, -0.01, 0.01)
        nn.init.uniform_(self.mean.weight, -0.01, 0.01)
        nn.init.uniform_(self.mean.bias, -0.01, 0.01)
        nn.init.uniform_(self.std.weight, -0.01, 0.01)
        nn.init.uniform_(self.std.bias, -0.01, 0.01)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        std = self.std(x)
        log_std = torch.tanh(std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x = normal.rsample()
        x = torch.tanh(x)
        return x

class RNDModel(nn.Module):
    def __init__(self, env):
        super(RNDModel, self).__init__()
        self.state_scale = torch.tensor(env.observation_space.high-env.observation_space.low).to('cuda')
        self.state_bias = (torch.tensor(env.observation_space.high-env.observation_space.low)/2).to('cuda')
        self.scaling_coef = 3
        self.obs = env.observation_space.shape[0]
        self.predictor = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 32)
        )
        
        self.target = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 32)
        )
        
        self.initial_weights = self.predictor.state_dict()
        self.soft_weights = self.predictor.state_dict()
        
        # Set target parameters as untrainable
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, next_obs):
        # normalization [-self.scaling_coef, self.scaling_coef]
        # next_obs = 2*self.scaling_coef*(next_obs-self.state_bias)/self.state_scale
        target_feature = self.target(next_obs)
        predict_feature = self.predictor(next_obs)
        return predict_feature, target_feature.detach()
    
    def rnd_bonus(self,s,normalize = True):
        predict, target = self(s)
        rnd_error = ((target+predict)**2).sum(axis=-1,keepdim=True)
        if normalize:
            rnd_error = (rnd_error-rnd_error.mean())/rnd_error.std()
        return rnd_error
    
    def rnd_reset(self):
        self.predictor.load_state_dict(self.initial_weights)
        
    def save_weights(self):
        self.soft_weights = self.predictor.state_dict()
    
    def soft_update(self):
        new_state_dict = {}
        for key in self.predictor.state_dict():
            new_state_dict[key] = 0.999*self.soft_weights[key].cuda()+0.001*self.predictor.state_dict()[key]
        self.predictor.load_state_dict(new_state_dict)

class Virtual_Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(env.observation_space.shape[0], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, env.action_space.shape[0])
        self.fc_logstd = nn.Linear(256, env.action_space.shape[0])
        
        # uniform initialize
        nn.init.uniform_(self.fc1.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc1.bias, -0.1, 0.1)
        nn.init.uniform_(self.fc2.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc2.bias, -0.1, 0.1)
        nn.init.uniform_(self.fc_mean.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc_mean.bias, -0.1, 0.1)
        nn.init.uniform_(self.fc_logstd.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc_logstd.bias, -0.1, 0.1)
        
        # action rescaling
        self.register_buffer(
                "action_scale", torch.tensor((env.action_space.high[0] - env.action_space.low[0]) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high[0] + env.action_space.low[0]) / 2.0, dtype=torch.float32)
        )
            

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def virtual_action(self, x, actor_action=None):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        if actor_action == None:
            y_t = torch.tanh(x_t)
            action = y_t * self.action_scale + self.action_bias
            log_prob = normal.log_prob(x_t)
            # Enforcing Action Bound
            log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        else:
            y_t = (actor_action-self.action_bias)/self.action_scale
            x_t = torch.atanh(y_t)
            log_prob = normal.log_prob(x_t)
            log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
            action = actor_action
        log_prob = log_prob.sum(-1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def get_entropy(self,x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        entropy = self.gaussian_entropy(std[0],std[1])
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return -log_prob
        return entropy
    
    def gaussian_entropy(self,std_x, std_y):
        entropy = 0.5 * torch.log(2 * torch.exp(torch.tensor(1)) * std_x**2 * std_y**2)
        return entropy