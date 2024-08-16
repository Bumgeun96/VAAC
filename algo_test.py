import argparse
from envs import (GymEnv,GymEnvDelayed)
import torch
import numpy as np
import random
from datetime import datetime
from algorithm.jta import jta_agent
from algorithm.PPO import PPO
from algorithm.ppo import ppo
from distutils.util import strtobool
from collections import defaultdict
import json
from plotlib import save_pickle
import wandb

ENV = "SparseHopper-v1"
# ENV = "SparseAnt-v1"
ENV = "SparseWalker2d-v1"
# ENV = "SparseHalfCheetah-v1"
# ENV = "HumanoidStandup-v1"
# ENV = "Humanoid-v1"
# ENV = "DelayedHopper-v1"
# ENV = "DelayedAnt-v1"
# ENV = "DelayedWalker2d-v1"
# ENV = "DelayedHalfCheetah-v1"
def parse_args():
    with open('parameters.json','r') as file:
        parameters = json.load(file)
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,default=10)

    # ENV hyperparameters
    parser.add_argument("--n_steps",type=int,default=parameters[ENV]['n_steps'])
    parser.add_argument("--n_total_steps",type=int,default=parameters[ENV]['n_total_steps'])
    parser.add_argument("--n_eval",type=int,default=parameters[ENV]['n_eval'])
    parser.add_argument('--n_iter_seed',type=int,default=parameters[ENV]['n_seeds'])
    if "Sparse" in parameters[ENV]['env']:
        parser.add_argument('--sparsity_level',type=int,default=int(parameters[ENV]['sparsity_level']))
    if "Delayed" in parameters[ENV]['env']:
        parser.add_argument('--delayed_level',type=int,default=int(parameters[ENV]['delayed_level']))
    
    # RL hyperparameters
    parser.add_argument('--gamma',type=float,default=parameters[ENV]['gamma'])
    parser.add_argument('--learning_start',type=int,default=parameters[ENV]['learning_start'])
    parser.add_argument("--algorithm", type=str, default=parameters[ENV]["algorithm"])

    #Encoder (Dynamics model or random encoder) hyperparameters
    parser.add_argument('--latent_size',type=int,default=parameters[ENV]['latent_size'])

    #RND hyperparameters
    parser.add_argument('--action_std_decay_freq',type=int,default=parameters[ENV]['action_std_decay_freq'])
    parser.add_argument('--action_std_init',type=float,default=parameters[ENV]['action_std_init'])
    parser.add_argument('--action_std_decay_rate',type=float,default=parameters[ENV]['action_std_decay_rate'])
    parser.add_argument('--min_action_std',type=float,default=parameters[ENV]['min_action_std'])
    parser.add_argument("--max_grad_norm",type=float,default=parameters[ENV]['max_grad_norm'])
    parser.add_argument("--clip_coef",type=float,default=parameters[ENV]['clip_coef'])
    parser.add_argument("--ent_coef",type=float,default=parameters[ENV]['ent_coef'])
    parser.add_argument("--vf_coef",type=float,default=parameters[ENV]['vf_coef'])
    parser.add_argument('--k_epochs',type=int,default=parameters[ENV]['k_epochs'])
    parser.add_argument("--update_timestep",type=int,default=parameters[ENV]['update_timestep'])
    parser.add_argument("--rnd_reward_scaling",type=float,default=parameters[ENV]['rnd_reward_scaling'])

    #Off-policy hyperparameters
    parser.add_argument('--buffer_size',type=int,default=parameters[ENV]['buffer_size'])
    parser.add_argument('--tau',type=float,default=parameters[ENV]['tau'])
    parser.add_argument('--batch_size',type=int,default=parameters[ENV]['batch_size'])
    parser.add_argument('--actor_lr',type=float,default=parameters[ENV]['actor_lr'])
    parser.add_argument('--critic_lr',type=float,default=parameters[ENV]['critic_lr'])
    parser.add_argument('--policy_frequency',type=int,default=parameters[ENV]['policy_frequency'])
    parser.add_argument('--target_network_frequency',type=int,default=parameters[ENV]['target_network_frequency'])

    #SAC hyperparameters
    parser.add_argument('--alpha',type=float,default=parameters[ENV]['alpha'])
    parser.add_argument("--auto_tune", type=lambda x:bool(strtobool(x)), default=bool(parameters[ENV]['auto_tune']), nargs="?", const=True)

    #VAAC hyperparameters
    parser.add_argument('--rnd_frequency',type=int,default=parameters[ENV]['rnd_frequency'])
    parser.add_argument('--rnd_reset',type=bool,default=bool(parameters[ENV]['rnd_reset']))
    parser.add_argument('--im_alpha',type=float,default=parameters[ENV]['im_alpha'])
    parser.add_argument('--im_beta',type=float,default=parameters[ENV]['im_beta'])
    parser.add_argument("--beta_scheduling", type=lambda x:bool(strtobool(x)), default=bool(parameters[ENV]['beta_scheduling']), nargs="?", const=True)
    parser.add_argument("--beta_init",type=float,default=parameters[ENV]['beta_init'])
    parser.add_argument("--beta_decay_freq",type=int,default=parameters[ENV]['beta_decay_freq'])
    parser.add_argument("--beta_decay_rate",type=float,default=parameters[ENV]['beta_decay_rate'])
    parser.add_argument("--min_beta",type=float,default=parameters[ENV]['min_beta'])

    #RE3 hyperparameters
    parser.add_argument("--re3_scale",type=float,default=parameters[ENV]['re3_scale'])
    
    #ICM hyperparameters
    parser.add_argument("--icm_beta",type=float,default=parameters[ENV]['icm_beta'])
    parser.add_argument("--icm_reward_scaling",type=float,default=parameters[ENV]['icm_reward_scaling'])

    #NovelD hyperparameters
    parser.add_argument("--noveld_alpha",type=float,default=parameters[ENV]['noveld_alpha'])
    parser.add_argument("--noveld_reward_scaling",type=float,default=parameters[ENV]['noveld_reward_scaling'])

    parser.add_argument('--algo',type=bool, default=False, help="the use of the proposed algorithm")
    args = parser.parse_args()
    return args

def train(env,eval_env, agent, n_episodes, max_step,training_steps,n_eval,policy_frequency):
    total_step = 0
    random_seed = random.randint(0,200)
    try:
        env.reset(seed = random_seed)
    except:
        env.reset()
    Eval = False
    if training_steps == None:
        n_episodes = 999999999
    returns = []
    for epi in range(1,n_episodes+1):
        state = env.reset()
        state = torch.tensor(state.reshape((1,state_size))[0],dtype=torch.float32)
        score = 0
        for step in range(1,max_step+1):
            total_step += 1
            action = agent.action(state).cpu().detach().numpy()
            action = np.clip(action*action_high, action_low, action_high)
            next_state, reward, done, truncated = env.step(action)
            next_state = next_state.reshape((1,state_size))
            next_state = torch.tensor(next_state[0],dtype=torch.float32)
            ################################################################
            # reward = 0
            ################################################################
            agent.store_experience(state, action, reward, next_state, done,total_step)
            agent.training(wandb)
            ################################################################
            # saving_buffer(state,action,next_state,total_step)
            ################################################################
            state = next_state
            score += reward
            if total_step % 1000 == 0:
                try:
                    torch.save(agent.actor.state_dict(),"./model/("+ENV+","+algo_args.algorithm+")policy.pt")
                    print('========================================')
                    now = datetime.now()
                    print('[',now.hour,':',now.minute,':',now.second,']','steps:',total_step + step)
                    Eval = True
                except:
                    torch.save(agent.policy.actor.state_dict(),"./model/("+ENV+","+algo_args.algorithm+")policy.pt")
                    print('========================================')
                    now = datetime.now()
                    print('[',now.hour,':',now.minute,':',now.second,']','steps:',total_step + step)
                    Eval = True
            if done or step>=1000:
                if Eval:
                    eval(eval_env,n_eval,total_step)
                    Eval = False
                break
        wandb.log({"episode reward":score},step=total_step)
        if total_step > training_steps:
            break
    return returns
        
def eval(eval_env,n_eval,total_step):
    try:
        eval_agent.actor.load_state_dict(torch.load("./model/("+ENV+","+algo_args.algorithm+")policy.pt"))
    except:
        eval_agent.policy.actor.load_state_dict(torch.load("./model/("+ENV+","+algo_args.algorithm+")policy.pt"))
    returns = 0
    step = 0
    for _ in range(n_eval):
        state=eval_env.reset().reshape((1,state_size))
        local_step = 0
        while True:
            step += 1
            local_step += 1
            state = torch.tensor(state[0],dtype=torch.float32).to('cuda')
            try:
                action = eval_agent.deterministic_act(state).cpu().detach().numpy()
            except:
                action = eval_agent.policy.deterministic_act(state).cpu().detach().numpy()
            action = np.clip(action*action_high, action_low, action_high)
            if local_step == 1:
                q_eval = agent.q_eval(state,action)
            next_state, reward, done, truncated= eval_env.step(action)
            state = next_state.reshape((1,state_size))
            returns += reward
            if done or local_step >= 1000:
                break
    returns /= n_eval
    step /= n_eval
    wandb.log({"returns":returns,"Q value":q_eval,"step":step},step=total_step)
    print('returns:',returns,'step:',step)
    return returns
        
def random_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def loading_algorithm(env,args):
    if args.algorithm == 'sac':
        from algorithm.sac import SAC_agent
        agent = SAC_agent(env,args)
    elif args.algorithm == 'vaac':
        from algorithm.vaac import VAAC_agent
        agent = VAAC_agent(env,args)
    elif args.algorithm == 'vaac(ac)':
        from algorithm.vaac_ac import VAAC_agent_ac
        agent = VAAC_agent_ac(env,args)
    elif args.algorithm == 'vaac_rnd':
        from algorithm.vaac_rnd import VAAC_rnd_agent
        agent = VAAC_rnd_agent(env,args)
    elif args.algorithm == 'rnd':
        from algorithm.sac_rnd import sac_rnd_agent
        agent = sac_rnd_agent(env,args)
    elif args.algorithm == 're3':
        from algorithm.re3 import re3_agent
        agent = re3_agent(env,args)
    elif args.algorithm == 'icm':
        from algorithm.sac_icm import SAC_icm_agent
        agent = SAC_icm_agent(env,args)
    elif args.algorithm == 'noveld':
        from algorithm.sac_noveld import sac_noveld_agent
        agent = sac_noveld_agent(env,args)
    elif args.algorithm == 'jta':
        agent = jta_agent(env,args)
    else:
        raise ValueError(args.algorithm)
    return agent


# import csv
# f = open('buffer.csv', 'w',newline='')
# writer = csv.writer(f)


# def saving_buffer(state,action,next_state,total_step):
#     state = state.tolist()
#     action = action.tolist()
#     next_state = next_state.tolist()
#     writer.writerow([state, action, next_state])
#     if total_step >= 500000:
#         f.close()
#         raise Exception("Save buffer")


if __name__ == "__main__":
    algo_args = parse_args()
    env_name = ENV
    print("algorithm:"+algo_args.algorithm)
    seeds = []
    for s in range(algo_args.n_iter_seed):
        seeds.append(10+10*s)
    return_values = defaultdict(list)
    returns = []
    seeds = [10,40,50]
    for seed in seeds:
        algo_args.seed = seed
        random_seed(seed)

        temp = {}
        for arg in vars(algo_args):
            temp[arg]=getattr(algo_args, arg)
        if "Sparse" in env_name:
            level = algo_args.sparsity_level
        elif "Delayed" in env_name:
            level = algo_args.delayed_level
        wandb.init(
            project = algo_args.algorithm,
            config = temp,
            name = ENV+"("+algo_args.algorithm+", level: "+str(level)+"),seed:"+str(algo_args.seed)
        )
        # wandb.init(
        #     project = "test",
        #     config = temp,
        #     name = ENV+"("+algo_args.algorithm+", level: "+str(algo_args.sparsity_level)+"),seed:"+str(algo_args.seed)
        # )

        if 'Delayed' in env_name:
            env = GymEnvDelayed(env_name.replace('Delayed',''),seed = seed,delay=algo_args.delayed_level)
            eval_env = GymEnvDelayed(env_name.replace('Delayed',''),seed = seed,delay=algo_args.delayed_level)
        else:
            env = GymEnv(env_name,seed = seed,sparsity_level = algo_args.sparsity_level)
            eval_env = GymEnv(env_name,seed = seed,sparsity_level = 1)
        action_high = env.action_space.high[0]
        action_low = env.action_space.low[0]
        action_bound = [action_low,action_high]
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]
        agent = loading_algorithm(env,algo_args)
        eval_agent = loading_algorithm(env,algo_args)
        train(env,
              eval_env,
              agent,
              999999999,
              algo_args.n_steps,
              algo_args.n_total_steps,
              algo_args.n_eval,
              algo_args.policy_frequency)
        wandb.finish()