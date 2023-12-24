


# ConSpec

## Information

The basic PPO code was taken from the repository: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail

The pycolab tasks codebase was taken from the repository: https://github.com/deepmind/deepmind-research/tree/master/tvt

## Installation
To create the environment install the packages from the requirements.txt file in a python 3.7 environment.



### Launching

To see conspecfunction in action with an underlying PPO agent on a 4-key task (fig. 4 of the paper), run: 
```
python -u main.py --pycolab_game key_to_door4  
```

### Using ConSpec on top of any other RL agent of choice

This function (conspecfunction) is meant to be a self-contained function and can be added to any RL agent of choice. 
To use this function in any  RL backbone, 3 simple steps need to be done (stated below). **A working example showing these lines of code added into an RL training loop is seen in main.py**


1. Load the "Conspec" subfolder into your codebase. This subfolder contains 5 files. 

2. In your main training script, import the following. All the relevant ConSpec functions and objects are contained in this class.
```
from Conspec.ConSpec import ConSpec
conspecfunction = ConSpec(args,   obsspace,  env.num_actions,  device)
```
   
3. In your RL training loop, simply add the 3 lines below.    
```
        obstotal, rewardtotal, recurrent_hidden_statestotal, actiontotal,  maskstotal  = rollouts.release()
        reward_intrinsic_extrinsic  = conspecfunction.do_everything(obstotal, recurrent_hidden_statestotal, actiontotal, rewardtotal, maskstotal)
        rollouts.storereward(reward_intrinsic_extrinsic)
```
the purpose of these last 3 lines is to: 
i. retrieve the current minibatch of trajectory (including its observations, rewards, hidden states, actions, masks)
ii. "do everything" that ConSpec needs to do internally for training, and output the intrinsic + extrinsic reward for the current minibatch of trajectories
iii. store this total reward in the memory buffer 

