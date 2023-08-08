import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from ContinuousGridworld.Continuous_GridWorld import ContinuousGridWorld
from algorithm.sac import SAC_agent
from algorithm.msac import MSAC_agent
from algorithm.iaac import IAAC_agent
from distutils.util import strtobool
from plotlib import plot_visiting, draw_env, visualization, save_pickle

import torch
import argparse
import json
import math

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,default=10)
    parser.add_argument('--n_steps',type=int,default=1000)
    parser.add_argument('--n_total_steps',type=int,default=300000)
    parser.add_argument('--n_iter_seed',type=int,default=30)
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
    parser.add_argument('--alpha',type=float,default=0.5)
    parser.add_argument("--auto_tune", type=lambda x:bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument('--mda_alpha',type=float,default=0.2)
    parser.add_argument('--im_alpha',type=float,default=0.2)
    parser.add_argument('--cliping_discriminator',type=float,default=0.0001)
    parser.add_argument('--d_step',type=int,default=1)
    parser.add_argument("--algo", type=bool, default=False)
    parser.add_argument("--algorithm", type=str, default='sac')
    parser.add_argument("--map",type=int, default=1)
    args = parser.parse_args()
    return args


def play(environment, agent, num_episodes=20, episode_length=1000, train=True,seed = 0):
    reward_per_episode = []
    state_scale = torch.tensor(environment.observation_space.high-environment.observation_space.low)
    state_bias = torch.tensor(environment.observation_space.high-environment.observation_space.low)/2
    returns = deque(maxlen=100)
    episode_return = 0
    visiting_times = []
    n_visitations = []
    total_step = 0
    for episode in range(num_episodes):
        timestep = 0
        terminal = False
        while timestep < episode_length and terminal != True:
            with torch.no_grad():
                current_state = 10*(torch.Tensor(environment.agent_location)-state_bias)/state_scale
                action = agent.action(current_state)
                action = np.array(action.to('cpu'))
                next_state, reward, terminal = environment.make_step(action,0)
            agent.count_visiting(next_state)
            next_state = torch.Tensor(next_state)
            next_state = 10*(next_state-state_bias)/state_scale
            timestep += 1
            total_step += 1
            if total_step % 10 == 0:
                n_visitation = agent.count_visitation()
                n_visitations.append(n_visitation)
            if total_step == 5000:
                visiting_time5k = agent.get_visiting_time()
                visiting_times.append(visiting_time5k)
                print(n_visitation)
            elif total_step == 50000:
                visiting_time50k = agent.get_visiting_time()
                visiting_times.append(visiting_time50k)
                print(n_visitation)
            elif total_step == 300000:
                visiting_time300k = agent.get_visiting_time()
                visiting_times.append(visiting_time300k)
                print(n_visitation)

            if train:
                agent.store_experience(current_state,action,reward,next_state,terminal)
                agent.training()

            if terminal or timestep >= episode_length:
                environment.reset()
            returns.append(reward)
        
        # table = agent.get_rnd_error()
        # v_table = agent.get_visiting_time()
        # visualization(environment,table,v_table)
        print('Training:',
              round(100*(seed*num_episodes*episode_length+total_step)/(num_episodes*episode_length*args.n_iter_seed),4),
              '%|',
              'Epi:',
              episode+1,
              '/',
              num_episodes,
              "|seed:",
              seed,
              "/",
              args.n_iter_seed)
        reward_per_episode.append(np.mean(returns))
    return reward_per_episode,visiting_times,n_visitations


def main(args):
    num_episodes = math.ceil(args.n_total_steps/args.n_steps)

    # Create environment
    env = ContinuousGridWorld(map = args.map)
    draw_env(env)
    
    ##################################
    v = []
    visiting_plots = []
    for i in range(args.n_iter_seed):
        args.seed += i
        if args.algorithm == 'sac':
            agent = SAC_agent(env,args)
        elif args.algorithm == 'msac':
            agent = MSAC_agent(env,args)
        elif args.algorithm == 'iaac':
            agent = IAAC_agent(env,args)
        reward_per_episode,visitings,visiting_plot = play(environment = env,
                                                            agent = agent,
                                                            num_episodes = num_episodes,
                                                            episode_length = args.n_steps,
                                                            train = True,
                                                            seed = i)
        v.append(visitings)
        visiting_plots.append(visiting_plot)
    save_pickle(visiting_plots, 'map:'+str(args.map)+","+args.algorithm + ',visitation_plot')
    v = np.array(v)
    visitings = np.sum(v,axis=0)
    fig, ax = plt.subplots(1, 3, figsize=(10, 4))
    per_name = ["5k", "50k", '300k']
    print(visitings)
    for i,t in enumerate(visitings):
        try:
            plot_visiting(ax[i],fig,env,t)
            ax[i].set_title(per_name[i], size=10)
        except:
            pass
    fig.savefig("map:"+str(args.map)+","+args.algorithm+",visiting_time.pdf")
    
    # Make learning curve
    plt.figure()
    plt.plot(range(1, num_episodes + 1), reward_per_episode, label="Q-learning")
    plt.xlabel("Episodes")
    plt.ylabel("Return per Episode")
    plt.legend()
    plt.savefig("learning_curve.pdf")


if __name__ == "__main__":
    with open('parameters.json','r') as file:
        parameters = json.load(file)
    args = parse_args()
    args.buffer_size = parameters["ContinuousGridWorld"]['buffer_size']
    args.n_steps = parameters["ContinuousGridWorld"]['n_steps']
    args.n_total_steps = parameters["ContinuousGridWorld"]['n_total_steps']
    args.n_iter_seed = parameters["ContinuousGridWorld"]['n_iter_seed']
    args.gamma = parameters["ContinuousGridWorld"]['gamma']
    args.tau = parameters["ContinuousGridWorld"]['tau']
    args.batch_size = parameters["ContinuousGridWorld"]['batch_size']
    args.learning_start = parameters["ContinuousGridWorld"]['learning_start']
    args.actor_lr = parameters["ContinuousGridWorld"]['actor_lr']
    args.critic_lr = parameters["ContinuousGridWorld"]['critic_lr']
    args.policy_frequency = parameters["ContinuousGridWorld"]['policy_frequency']
    args.noise_clip = parameters["ContinuousGridWorld"]['noise_clip']
    args.alpha = parameters["ContinuousGridWorld"]['alpha']
    args.auto_tune = bool(parameters["ContinuousGridWorld"]['auto_tune'])
    args.mda_alpha = parameters["ContinuousGridWorld"]['mda_alpha']
    args.im_alpha = parameters["ContinuousGridWorld"]['im_alpha']
    args.cliping_discriminator = parameters["ContinuousGridWorld"]['cliping_discriminator']
    args.d_step = int(parameters["ContinuousGridWorld"]['d_step'])
    args.algorithm = parameters["ContinuousGridWorld"]["algorithm"]
    args.map = parameters["ContinuousGridWorld"]["map"]
    args.algo = False
    print("algorithm:"+args.algorithm)
    print("map:"+str(args.map))
    main(args)