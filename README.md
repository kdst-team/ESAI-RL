# ESAI-RL

**version 0.2**

- implement RL algorithm (Q-learning and Deep Q-Network (DQN) function )



#### Dependencies 

------

- python3 >= 3.6
- pytorch >= 1.2

- open-ai gym == 0.15.4

- numpy ==1.19.2

- matplotlib ==3.1.1




### 1. ESAI-RL Q-learning 

------

- Most reinforcement learning frameworks do not implement Q-learning algorithms.

- ESAI aims to simplicity of usage by modularizing the Q-learning algorithm
- ESAI Q-learning class can use only __discrete__ environment of openai-gym [^1]

![discrete_env](C:\Users\KimYujin\Desktop\KIST\Reinforcement Learning\RL progress\discrete_env.png)

[^1]: current openai-gym version support Taxi-v3 and you can use ESAI Q-learning class on Taxi-v3



#### Paper Reference

- Q-learning paper : [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602.pdf)



#### Q-learning class & method

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

#### Usage

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





### 2. ESAI-RL DQN (Deep Q-Network)

- ESAI-RL DQN aims to simplicity of usage by modularizing the DQN method on continuous environment.

- It is compatible with pytorch and allows you to create any DQN network you want.

  

#### Paper Reference

- Deep Q-Network paper : https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf




#### Deep Q-Network method

------

| Method                                                       | Explanation     |
| ------------------------------------------------------------ | --------------- |
| `esai.dqn` *(env = env, net = QNetwork, pretrained = "./checkpoint.pth", render = True, n_episodes = 2000, max_t= 1000, eps_start = 0.9 ,eps_end = 0.01 ,eps_decay = 0.995, cp = 100)* | train DQN agent |

##### Parameters

- **env** : *{"BipedalWalker-v2" etc..}*

  - Box-2d environment  on gym : https://github.com/openai/gym/tree/master/gym/envs/box2d

  - BipedalWalker environment state-space, action-space (*continuous*)

    - state : *Box(24,)*
    - action : Box(4,)

- **net** : *default="QNetwork"*

  - Class name of custom model

  - ex) **model.py**  - class name : *QNetwork*

    ```python
    class QNetwork(nn.Module):
        def __init__(self, state_size, action_size, seed, fc1_units=512, 
            super(QNetwork, self).__init__()
                     
        def forward(self, state):
    ```

- **pretrained** : *str, default =""*

  - Path where checkpoint file is located and training by checkpoint, not initial state

- **render**  :  *bool, default="False"*

  - Rendering (whether or not to visualize environment) option on gym

- **n_episodes** : *int, default = 2000*

  - the number of episodes

- **max_t** : *int, default = 1000*

  - Maximum number of time steps per episode

- **eps_start** : *float, default = 0.9*

  - Starting value of epsilon, for epsilon-greedy action selection

- **eps_end** : *float, default = 0.01*

  - Minimum value of epsilon

- **eps_decay** : *float, default 0.995*

  - Multiplicative factor (per episode) for decreasing epsilon 

- **cp** : *int, default = 100*

  - period of saving checkpoint

##### Return

- rewards : *list, len : (n_episodes)*
  - total reward list 



#### Usage - 1. basic esai_dqn 

------

```python
import gym
import numpy
import esai_rl as esai
import matplotlib.pyplot as plt
import torch
import numpy as np

#customized model architecture in model.py 
from model import QNetwork

#define environment in gym
env=gym.make('BipedalWalker-v2')

#start to train using esai-dqn
scores =esai.dqn(env=env,net=QNetwork,pretrained="",render=False,n_episodes=2000, max_t=1000, eps_start=0.9, eps_end=0.01, eps_decay=0.995,cp=100)

```



#### custom DQN network

------

You can easily define a custom architecture for DQN in **model.py** 

- you need to define **environment state space** as input size, **action space** as output size

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """DQN network"""
    def __init__(self, state_size, action_size, seed, fc1_units=512, fc2_units=256):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
```



#### Usage - 2. draw reward graph

------

`rolling_window_mean` and `plot_graph` is customized function for drawing reward graph.

In `rolling_window_mean` function, calculate moving average on total rewards, In `plot_graph` function, plot the graph using matplot library.

```python
import gym
import numpy
import esai_rl as esai
import matplotlib.pyplot as plt
import torch
import numpy as np
from model2 import QNetwork

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

scores = esai.dqn(env=env,net=QNetwork,pretrained="",render=False,n_episodes=2000, max_t=1000, eps_start=0.9, eps_end=0.01, eps_decay=0.995,cp=100)

scores = torch.FloatTensor(scores)
rolling_mean = rolling_window_mean(scores,100,1)
plot_graph(rolling_mean)

```



#### result

- print Average score, aver FPS, max step of agent per 100 episodes
- graph using moving average

![image-20201102144637934](C:\Users\KimYujin\AppData\Roaming\Typora\typora-user-images\image-20201102144637934.png)

<img src="C:\Users\KimYujin\AppData\Roaming\Typora\typora-user-images\image-20201102161732356.png" alt="image-20201102161732356" style="zoom: 67%;" />