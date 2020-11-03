import gym
import numpy
import esai_rl as esai
import matplotlib.pyplot as plt
import torch
import numpy as np 
from model import QNetwork

def rolling_window_mean(x, window_size, step_size=1):
    rolling_mean = []
    result = x.unfold(0,window_size,step_size)
    for i in range(result.shape[0]):
       mean = torch.mean(result[i])
       rolling_mean.append(mean.item())
    return rolling_mean

def plot_graph(scores):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()    

env=gym.make('BipedalWalker-v2')
#net = QNetwork(env.observation_space.shape[0], env.action_space.shape[0],10)

scores = esai.dqn(env=env,net = QNetwork,pretrained="",render=False,n_episodes=10000, max_t=1000, eps_start=0.9, eps_end=0.01, eps_decay=0.995,cp=100)

scores = torch.FloatTensor(scores)
rolling_mean = rolling_window_mean(scores,100,1)
plot_graph(rolling_mean)
