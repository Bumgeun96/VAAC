import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import numpy as np

class re3(nn.Module):
    def __init__(self,input_size,latent_size,k=3):
        super().__init__()
        self.k = k
        self.encoder = nn.Sequential(
            nn.Linear(input_size,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,latent_size),
            nn.LayerNorm(latent_size)
        )

    def forward(self,state):
        feature_vector = self.encoder(state)
        return feature_vector

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
    
class rnd(nn.Module):
    def __init__(self,env,latent_size,input_size = None):
        super().__init__()
        if input_size == None:
            input_size = env.observation_space.shape[0]
        self.model = nn.Sequential(
            nn.Linear(input_size,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,latent_size)
        )
        self.target = nn.Sequential(
            nn.Linear(input_size,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,latent_size)
        )

    def get_reward(self,state):
        reward = ((self.target(state).detach()-self.model(state))**2).sum(dim=1,keepdims=True)
        return reward
    
class dynamics_model(nn.Module):
    def __init__(self, env, latent_size):
        super().__init__()
        self.en1 = nn.Linear(env.observation_space.shape[0],256)
        self.en2 = nn.Linear(256,latent_size)

        self.de1 = nn.Linear(latent_size+env.action_space.shape[0],512)
        self.de2 = nn.Linear(512,512)
        self.de3 = nn.Linear(512,latent_size)
    
    def forward(self,state,action=None):
        z = self.encoder(state)
        if action==None:
            return z
        else:
            z_prime = self.decoder(z,action)
            return z_prime

    def encoder(self, state):
        z = F.relu(self.en1(state))
        z = self.en2(z)
        return z
    
    def decoder(self, z, action):
        z = torch.cat([z, action], 1)
        z = F.relu(self.de1(z))
        z = F.relu(self.de2(z))
        z = self.de3(z)
        return z

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
    
    def q_eval(self,state,action):
        x = torch.cat([state,action])
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
        nn.init.uniform_(self.fc1.weight, -0.01, 0.01)
        nn.init.uniform_(self.fc1.bias, -0.01, 0.01)
        nn.init.uniform_(self.fc2.weight, -0.01, 0.01)
        nn.init.uniform_(self.fc2.bias, -0.01, 0.01)
        nn.init.uniform_(self.fc_mean.weight, -0.01, 0.01)
        nn.init.uniform_(self.fc_mean.bias, -0.01, 0.01)
        nn.init.uniform_(self.fc_logstd.weight, -0.01, 0.01)
        nn.init.uniform_(self.fc_logstd.bias, -0.01, 0.01)
        
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

class RNDModel(nn.Module):
    def __init__(self, env, latent_size):
        super(RNDModel, self).__init__()
        self.obs = env.observation_space.shape[0]
        self.obs = latent_size
        self.predictor = nn.Sequential(
            nn.Linear(self.obs, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 32)
        )
        
        self.target = nn.Sequential(
            nn.Linear(self.obs, 256),
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

    def virtual_action(self, x):
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
    
    def log_prob(self,action):
        y_t = (action - self.action_bias)/self.action_scale
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return log_prob

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
        # return std.sum()
        # return -log_prob
        return entropy
    
    def gaussian_entropy(self,std_x, std_y):
        entropy = 0.5 * torch.log(2 * torch.exp(torch.tensor(1)) * std_x**2 * std_y**2)
        return entropy
    
    
class PPO_ActorCritic(nn.Module):
    def __init__(self, env, state_dim, action_dim, action_std_init):
        super(PPO_ActorCritic, self).__init__()
        self.device = torch.device('cpu')
        if(torch.cuda.is_available()): 
            self.device = torch.device('cuda:0') 
            torch.cuda.empty_cache()
            
        self.action_dim = action_dim
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(self.device)
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
        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(self.device)

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

    def deterministic_act(self, state):
        action_mean = self.actor(state)
        action = torch.tanh(action_mean)* self.action_scale + self.action_bias
        return action.detach()
    
    def evaluate(self, state, action):
        action_mean = self.actor(state)
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(self.device)
        dist = MultivariateNormal(action_mean, cov_mat)
        
        # For Single Action Environments.
        if self.action_dim == 1:
            action = action.reshape(-1, self.action_dim)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy
    

class PPO_AC(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            self.layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            self.layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, np.prod(envs.action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.action_space.shape))[0])

    def get_value(self, x):
        return self.critic(x)
    
    def layer_init(self,layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        print(action_std)
        probs = torch.distributions.Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(), probs.entropy().sum(), self.critic(x)
    
    def deterministic_act(self, x):
        action_mean = self.actor_mean(x)
        return action_mean.detach()

class icm(nn.Module):
    def __init__(self,env,latent_size):
        super().__init__()

        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high[0] - env.action_space.low[0]) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high[0] + env.action_space.low[0]) / 2.0, dtype=torch.float32)
        )

        self.encoder = nn.Sequential(
            nn.Linear(env.observation_space.shape[0],256),
            nn.ReLU(),
            nn.Linear(256,latent_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size+env.action_space.shape[0],512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,latent_size)
        )
        self.inverse_model = nn.Sequential(
            nn.Linear(2*latent_size,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,env.action_space.shape[0]),
            nn.Tanh()
        )

    def action_pred(self,state,next_state):
        z = self.encoder(state)
        z_prime = self.encoder(next_state)
        action = self.inverse_model(torch.cat([z,z_prime],dim=1))
        action = action * self.action_scale + self.action_bias
        return action
    
    def forward_model(self,state,action):
        z = self.encoder(state)
        z_prime = self.decoder(torch.cat([z,action],dim=1))
        return z_prime
    
    def forward(self):
        raise NotImplementedError