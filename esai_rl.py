import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd
import time

from dqn_agent import Agent

class QLearning:
    def __init__(self, env, alpha, gamma, epsilon):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.n_state = env.observation_space.n
        self.n_action = env.action_space.n
        self.q_table = np.zeros((self.n_state, self.n_action))
        self.S = np.arange(self.n_state)
        self.A = np.arange(self.n_action)
        self.rList = []
        self.culList=[]


    def egreedy_policy(self,q_table, state, epsilon):
        # Get a random number from a uniform distribution between 0 and 1,
        # if the number is lower than epsilon choose a random action
        if np.random.random() < epsilon:
            #print (epsilon)
            return np.random.choice(self.A)
            #return self.env.action_space.sample()
        # Else choose the action with the highest value
        else:
            #print(np.argmax(self.q_table[state,:]))
            return np.argmax(self.q_table[state,:])


    def run_episode(self,render=False,n_episode=2000):
        #rList=[]
        #ulList=[]
        #self.rList.clear()
        #self.culList.clear()
        successRate=[]
        cul_reward=0
        total_reward=0
        for i in range(n_episode):
            state = self.env.reset()
            total_reward = 0
            done = False

        # While episode is not over
            while not done:
             # Choose action
                #action = self.egreedy_policy(self.q_table, state, self.epsilon)
                action = np.argmax(self.q_table[state,:] + np.random.randn(1,self.n_action)*(1./(i+1)))
                next_state, reward, done, info = self.env.step(action)
                td_target = reward + (self.gamma * np.max(self.q_table[next_state]))
                td_error = td_target - self.q_table[state][action]
                self.q_table[state][action] += self.alpha * td_error
                #self.q_table[state][action]=(1-self.alpha)*self.q_table[state][action]+self.alpha*(reward+self.gamma*np.max(self.q_table[next_state]))
                # Update state
                total_reward+=reward
                cul_reward+=reward
                state = next_state
                if render:
                    self.env.render()
                #env.render()
            print("Episode : ",i ,"State: ",state, "Action: ",action, "Reward: ",reward)
            self.rList.append(total_reward)
            self.culList.append(cul_reward)
            #successRate.append(sum(rList)/(i+1))
        print (total_reward)
        print (self.q_table)
        #print("success rate :",successRate[-1])


    def plot_reward(self):
        fig=plt.figure()
        #plt.title('{} reward'.format(self.env))
        ax1=fig.add_subplot(211)
        ax2=fig.add_subplot(212)

        ax1.plot(range(len(self.rList)),self.rList,'r')
        ax2.plot(range(len(self.culList)),self.culList,'b')

        ax1.set_xlabel('episode')
        ax1.set_ylabel('reward')

        ax2.set_xlabel('episode')
        ax2.set_ylabel('total_reward')
        plt.show()
            #if reward==1.0:
                #print("%dth episode finished and reward is %.1f" %(_,reward) )
                #break
            #env.render()



def dqn(env= None,net = None,pretrained=None,n_episodes=10000, max_t=1000, eps_start=0.9, eps_end=0.01, eps_decay=0.995,render=False, cp = 100):

    print('State shape: ', env.observation_space.shape)
    print('Number of actions: ', env.action_space.shape)
    agent = Agent(state_size=env.observation_space.shape[0], action_size=env.action_space.shape[0],seed=10, net=net,pretrained = pretrained)
    state = env.reset()
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    fps_list=[]
    eps = eps_start                    # initialize epsilon
    
    #if "pth" in pretrained:
        #agent.qnetwork_local.load_state_dict(torch.load(pretrained))

    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        fps = 0
        start_time = time.time()
        for t in range(max_t):
            action = agent.act(state, eps)
            if render: 
               env.render()
            fps = fps+1
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        end_time = time.time()
        epi_fps = (fps / (end_time - start_time))
        #print("fps : ",fps)
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        fps_list.append(epi_fps)
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}\tAverage FPS : {:.2f} max_t:{}'.format(i_episode, np.mean(scores_window),np.mean(fps_list),t), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}\tAverage FPS : {:.2f}'.format(i_episode, np.mean(scores_window),np.mean(fps_list)))
        if np.mean(scores_window)>=200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'best_checkpoint.pth')
            #break
        if i_episode % cp == 0:
            torch.save(agent.qnetwork_local.state_dict(),'{}_cp.pth'.format(i_episode))
    pd.Series(scores).to_csv('rewards.csv', index=None)
     
    return scores

