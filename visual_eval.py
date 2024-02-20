import torch
import numpy
import json
import time
import os
import argparse
import imageio
from PIL import Image
import numpy as np
from rl_utils.network import Actor, PPO_ActorCritic
from envs import (GymEnv,GymEnvDelayed)

with open('parameters.json','r') as file:
    parameters = json.load(file) 

ENV = ["SparseHopper-v1","SparseAnt-v1","SparseWalker2d-v1","SparseHalfCheetah-v1",
       "HumanoidStandup-v1","Humanoid-v1",
       "DelayedAnt-v1"]

parser = argparse.ArgumentParser()
parser.add_argument('--speed',type=float,default=0)
parser.add_argument('--type',type=str,default='deterministic')
parser.add_argument('--env',type=str,default='SparseAnt-v1')
parser.add_argument('--epi_length',type=int,default=1000)
parser.add_argument('--video_record',type=str,default='False')
args = parser.parse_args()

if args.video_record == 'True':
    delay_time = 0
else:
    delay_time = max(-0.01*(args.speed-4),0)


env_set = set(args.env)
length = 0
for e in ENV:
    if length <= len(set(e) & env_set):
        length = len(set(e) & env_set)
        environment = e

    
if 'Delayed' in environment:
    env = GymEnvDelayed(environment.replace('Delayed',''),delay=20)
else:
    env = GymEnv(environment)

print('==============================================')
print('Environment:',environment)
print('Algorithm:',parameters[environment]["algorithm"])
print('==============================================')


if parameters[environment]["algorithm"] == 'sac' or parameters[environment]["algorithm"] == 'vaac':
    agent = Actor(env)
elif parameters[environment]["algorithm"] == 'rnd':
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = PPO_ActorCritic(env, state_dim, action_dim, 1)

networks = os.listdir("./model")
networks_env = []
networks_env_algo = []
for network in networks:
    if environment in network:
        networks_env.append(network)
for t in networks_env:
    if parameters[environment]["algorithm"] in t:
        networks_env_algo.append(t)
net  = networks_env_algo[0]
    

if parameters[environment]["algorithm"] == 'sac' or parameters[environment]["algorithm"] == 'vaac':
    agent.load_state_dict(torch.load("./model/"+net))
    print('Model:',net)
    print('Network loaded')
elif parameters[environment]["algorithm"] == 'rnd':
    agent.actor.load_state_dict(torch.load("./model/"+net))
    print('Model:',net)
    print('Network loaded')

images = []
video_name = './video/'+environment+'('+parameters[environment]["algorithm"]+')'

observation = env.reset()
with torch.no_grad():
    total_reward = 0
    for t in range(args.epi_length):  # 렌더링을 원하는 프레임 수로 조절
        if args.video_record == 'True':
            img = env.render(mode='rgb_array')
            new_size = (1536, 1536)
            img = np.array(Image.fromarray(img).resize(new_size))
            images.append(img)
        else:
            env.render()
        if parameters[environment]["algorithm"] == 'sac' or parameters[environment]["algorithm"] == 'vaac':
            if args.type == 'deterministic' or args.video_record == 'True':
                __,__,action = agent.get_action(torch.tensor(observation,dtype=torch.float32))
            elif args.type == 'stochastic' and args.video_record == 'False':
                action,log,__ = agent.get_action(torch.tensor(observation,dtype=torch.float32))
        elif parameters[environment]["algorithm"] == 'rnd':
            action = agent.deterministic_act(torch.tensor(observation,dtype=torch.float32)).cpu().detach().numpy()
        observation, reward, done, __ = env.step(action)
        print('Timestep:',t+1)
        print('Reward:',reward)
        total_reward += reward
        # 에피소드 종료 시 초기화
        if done or t+1 == args.epi_length:
            print('Total reward:',total_reward)
            if args.video_record == 'True':
                break
            total_reward = 0
            observation = env.reset()
        time.sleep(delay_time)

if args.video_record == 'True':
    imageio.mimsave(video_name+'(reward: '+str(total_reward)+').mp4', images)
    print('Video Saved!!!!')