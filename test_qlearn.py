import gym
import numpy
import esai_rl as esai

env=gym.make('FrozenLake-v0')
agent=esai.QLearning(env,alpha=0.85,gamma=0.99,epsilon=0.2)
agent.run_episode(render=False,n_episode=2000)
agent.plot_reward()
