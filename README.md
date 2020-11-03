## ESAI Q-learning  

__version 0.1__

- Most reinforcement learning frameworks do not implement Q-learning algorithms.

- ESAI aims to simplicity of usage by modularizing the Q-learning algorithm
- ESAI Q-learning class can use only __discrete__ environment of openai-gym [^1]

![discrete_env](C:\Users\KimYujin\Desktop\KIST\Reinforcement Learning\RL progress\discrete_env.png)

[^1]:current openai-gym version support Taxi-v3 and you can use ESAI Q-learning class on Taxi-v3



## Dependencies

- python : 3.6.8
- openai-gym : 0.15.4
- numpy : 1.16.5
- matplotlib : 3.1.1



## Paper Reference

- Q-learning paper : [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602.pdf)





## Q-learning class & method

| Class                                      |      |
| ------------------------------------------ | ---- |
| `esai.Qlearning (env,alpha,gamma,epsilon)` |      |

__parameters__

* __env__ : {"Roulette-v0","NChain-v0","FrozenLake-v0","CliffWalking-v0","FrozenLake8x8-v0","Taxi-v3"}
  * choose gym environment already registered in open-gym
* __alpha__ : *float*
  * learning rate

* __gamma__ : *float*
  * discount factor

* __epsilon__ : *float*

  * fraction of entire training period over which the exploration rate is annealed

    

| Method                                      | Explanation                          |
| ------------------------------------------- | ------------------------------------ |
| `run_episode(render=False, n_episode=2000)` | train Q-table and return reward list |

__parameters__

* __render__ : *bool , default=False*
  * Whether or not to visualize environment

* __n_episode__ = *int, default=2000* 
  * the number of episodes



| method          | explanation                                                  |
| --------------- | ------------------------------------------------------------ |
| `plot_reward()` | return reward graph per episodes and cumulative rewards graph |

__graph example__

![FrozenLake-v0 reward graph](C:\Users\KimYujin\Desktop\KIST\Reinforcement Learning\RL progress\frozenLake-v0.png)

## Usage

```python
import gym
import esai

env=gym.make('FrozenLake-v0')
agent=esai.QLearning(env,alpha=0.85,gamma=0.99,epsilon=0.2)
#env,alpha,gamma,epsilon is hyper parameter
agent.run_episode(render=False,n_episode=2000)
# n_episode is hyper parameter
agent.plot_reward()
# if you want to check graph, use plot_reward() method
```

