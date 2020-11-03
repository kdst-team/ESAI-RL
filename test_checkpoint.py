import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import time
import datetime

from dqn_agent import Agent
from model import QNetwork

env = gym.make('BipedalWalker-v2')
env.seed(10)
agent = Agent(state_size=env.observation_space.shape[0], action_size=env.action_space.shape[0], seed=0,net=QNetwork, pretrained="./5000_cp.pth")

#agent.qnetwork_local.load_state_dict(torch.load("./5000_cp.pth"))

start_time = 0
end_time = 0
isrange=0
def test(epi_num):
   fps=0
   score = 0.
   state = env.reset()
   max_time_end = int(time.time())+5
   
   start_time = int(time.time())
   while True:
      if time.time() > max_time_end:
         end_time = int(time.time())
         #print("end time : ",int(time.time()))
         isrange=1
         break
      env.render()
      action = agent.act(state)
      next_state, reward, done, _ = env.step(action)
      fps = fps+1
      score = score + reward
      state = next_state
      if done:
         end_time = int(time.time())
         if (end_time - start_time) >=5:
            isrange = 1
         else:
            isrange = 0
         break
   return fps,reward,isrange,end_time,start_time


for i in range(0,100):
    #print(" ---------{} test_episode ------------".format(i))  
    fps,rwd,tf,end_time,start_time = test(i)
    if i >=2 and tf is 1:
        print(" --------------{} test_episode ---------------".format(i-2))
        print("start : ", datetime.datetime.fromtimestamp(start_time))
        print("fps during 5 sec : {}".format(fps/(end_time-start_time),rwd))
        print("end : ", datetime.datetime.fromtimestamp(end_time))
env.close()
