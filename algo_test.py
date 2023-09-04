import argparse
import gym
import sparse_gym_mujoco
from gymnasium.utils.save_video import save_video
import torch
import numpy as np
import random
from datetime import datetime
import pickle
from algorithm.sac import SAC_agent
from algorithm.vaac import VAAC_agent
from algorithm.PPO import PPO
from distutils.util import strtobool
from collections import defaultdict
import json

ENV = "Hopper-v1"
ENV = "SparseHopper-v1"
# ENV = "SparseHalfCheetah-v1"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,default=10)
    parser.add_argument('--n_iter_seed',type=int,default=20)
    parser.add_argument('--buffer_size',type=int,default=int(1e6))
    parser.add_argument('--gamma',type=float,default=0.99)
    parser.add_argument('--tau',type=float,default=0.005)
    parser.add_argument('--batch_size',type=int,default=256)
    parser.add_argument('--learning_start',type=int,default=1e3)
    parser.add_argument('--actor_lr',type=float,default=3e-4)
    parser.add_argument('--critic_lr',type=float,default=1e-3)
    parser.add_argument('--policy_frequency',type=int,default=2)
    parser.add_argument('--target_network_frequency',type=int,default=1)
    parser.add_argument('--noise_clip',type=float,default=0.5)
    parser.add_argument('--action_std_decay_freq',type=int,default=250000)
    parser.add_argument('--action_std_init',type=float,default=0.6)
    parser.add_argument('--k_epochs',type=int,default=80)
    parser.add_argument('--action_std_decay_rate',type=float,default=0.05)
    parser.add_argument('--min_action_std',type=float,default=0.1)
    parser.add_argument('--im_alpha',type=float,default=0.2)
    parser.add_argument('--im_beta',type=float,default=0.2)
    parser.add_argument('--alpha',type=float,default=0.5)
    parser.add_argument("--auto_tune", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument('--algo',type=bool, default=False, help="the use of the proposed algorithm")
    parser.add_argument("--algorithm", type=str, default='sac')
    parser.add_argument("--n_total_steps",type=int,default=1000000)
    args = parser.parse_args()
    return args

def train(env, agent, n_episodes, max_step,training_steps,n_eval):
    total_step = 0
    random_seed = random.randint(0,200)
    try:
        env.reset(seed = random_seed)
    except:
        env.reset()
    Eval = False
    if training_steps == None:
        n_episodes = 999999999
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
            agent.store_experience(state, action, reward, next_state, done)
            agent.training()
            state = next_state
            score += reward
            if total_step % 1000 == 0:
                try:
                    torch.save(agent.actor.state_dict(),"./model/("+ENV+")policy.pt")
                    print('========================================')
                    now = datetime.now()
                    print('[',now.hour,':',now.minute,':',now.second,']','steps:',total_step + step)
                    Eval = True
                except:
                    torch.save(agent.policy.actor.state_dict(),"./model/("+ENV+")policy.pt")
                    print('========================================')
                    now = datetime.now()
                    print('[',now.hour,':',now.minute,':',now.second,']','steps:',total_step + step)
                    Eval = True
            if done or truncated:
                # try:
                #     print('returns:',score[0],'step:',step)
                # except:
                #     print('returns:',score,'step:',step)
                if Eval:
                    returns = eval(n_eval)
                    return_values[algo_args.seed].append(returns)
                    Eval = False
                    with open('./figures/returns_'+ENV+'.pickle',"wb") as fw:
                        pickle.dump(return_values,fw)
                break
        if total_step > training_steps:
            break
        
def eval(n_eval):
    eval_env = gym.make(env_name)
    try:
        eval_agent.actor.load_state_dict(torch.load("./model/("+ENV+")policy.pt"))
    except:
        eval_agent.policy.actor.load_state_dict(torch.load("./model/("+ENV+")policy.pt"))
    returns = 0
    step = 0
    for _ in range(n_eval):
        state=eval_env.reset().reshape((1,state_size))
        while True:
            step += 1
            state = torch.tensor(state[0],dtype=torch.float32).to('cuda')
            try:
                action = eval_agent.deterministic_act(state).cpu().detach().numpy()
            except:
                action = eval_agent.policy.deterministic_act(state).cpu().detach().numpy()
            action = np.clip(action*action_high, action_low, action_high)
            next_state, reward, done, truncated= eval_env.step(action)
            next_state = next_state.reshape((1,state_size))
            state = next_state
            returns += reward
            if done or truncated:
                # save_video(eval_env.render(),
                #            "videos",
                #            fps=eval_env.metadata["render_fps"],
                #            step_starting_index=step-1)
                break
    returns /= n_eval
    step /= n_eval
    print('returns:',returns,'step:',step)
    eval_env.close()
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
        agent = SAC_agent(env,args)
    elif args.algorithm == 'vaac':
        agent = VAAC_agent(env,args)
    elif args.algorithm == 'ppo':
        args.action_std_decay_freq =int(args.n_total_steps/7)
        agent = PPO(env,args)
    return agent

if __name__ == "__main__":
    algo_args = parse_args()
    env_name = ENV
    with open('parameters.json','r') as file:
        parameters = json.load(file)
        
    algo_args.buffer_size = parameters[ENV]['buffer_size']
    algo_args.gamma = parameters[ENV]['gamma']
    algo_args.tau = parameters[ENV]['tau']
    algo_args.batch_size = parameters[ENV]['batch_size']
    algo_args.learning_start = parameters[ENV]['learning_start']
    algo_args.actor_lr = parameters[ENV]['actor_lr']
    algo_args.critic_lr = parameters[ENV]['critic_lr']
    algo_args.policy_frequency = parameters[ENV]['policy_frequency']
    algo_args.noise_clip = parameters[ENV]['noise_clip']
    algo_args.k_epochs = parameters[ENV]['k_epochs']
    algo_args.action_std_decay_freq = parameters[ENV]['action_std_decay_freq']
    algo_args.action_std_init= parameters[ENV]['action_std_init']
    algo_args.action_std_decay_rate= parameters[ENV]['action_std_decay_rate']
    algo_args.min_action_std= parameters[ENV]['min_action_std']
    algo_args.im_alpha = parameters[ENV]['im_alpha']
    algo_args.im_beta = parameters[ENV]['im_beta']
    algo_args.algorithm = parameters[ENV]["algorithm"]
    algo_args.alpha = parameters[ENV]['alpha']
    algo_args.auto_tune = bool(parameters[ENV]['auto_tune'])
    algo_args.n_steps = parameters[ENV]['n_steps']
    algo_args.n_total_steps = parameters[ENV]['n_total_steps']
    print("algorithm:"+algo_args.algorithm)
    env = gym.make(env_name)
    action_high = env.action_space.high[0]
    action_low = env.action_space.low[0]
    action_bound = [action_low,action_high]
    seeds = [10,20,30,40,50]
    return_values = defaultdict(list)
    for seed in seeds:
        algo_args.seed = seed
        random_seed(seed)
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]
        agent = loading_algorithm(env,algo_args)
        eval_agent = loading_algorithm(env,algo_args)
        train(env,
              agent,
              100000,
              parameters[ENV]['n_steps'],
              parameters[ENV]['n_total_steps'],
              parameters[ENV]['n_eval'])
    env.close()