


# ConSpec

## Information

The basic PPO code was taken from the repository: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail

The pycolab tasks codebase was taken from the repository: https://github.com/deepmind/deepmind-research/tree/master/tvt

## Installation
To create the environment install the packages from the requirements.txt file



### Launching

To see conspecfunction in action with an underlying PPO agent, run: 
```
python -u main.py  --algo ppo  --factorC 1.0 --use-gae --lr 2e-4 --clip-param 0.08 --value-loss-coef 0.5 --num-processes 16  --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.02 --lrCL 20e-4 --choiceCLparams 0 --seed 80001 --expansion 5000 --pycolab_apple_reward_min 0. --pycolab_apple_reward_max 0. --pycolab_final_reward 1. --factorR 0.2  --head 8
```

This function (conspecfunction) is meant to be a self-contained function and can be added to any RL agent of choice. 
To use this function in another RL backbone, 4 steps need to be done. These 3 steps are written out for convenience below. They can be seen in working order in mainTVTofficial.py.

1. Load the "Conspec" subfolder into your codebase. This sbufolder contains the 4 functions needed to train ConSpec. 

2. In your main training script, import the following:
```
from Conspec.moduleConSpec import moduleCL

from Conspec.conspecfunction import conspecfunctionx
```
3. In your script, define the following, and set up the appropriate arguments: 
```
moduleuse = moduleCL(...)

moduleuse.to(device)

conspecfunction = conspecfunctionx(...)
```
4. To train ConSpec using these functions, run the following, :
```
obstotal, rewardtotal, recurrent_hidden_statestotal, actiontotal,  maskstotal  = rollouts.release()

rewardtotalintrisic  = conspecfunction.storeandupdate(obstotal, recurrent_hidden_statestotal, actiontotal,  rewardtotal, maskstotal, j) # where j is the # of training iterations so far

rollouts.storeintrinsic(rewardtotalintrisic)
```
The 1st line collects the current minibatch. The 2nd line stores it within the conspecfunction object and runs 1 gradient update on ConSpec. Note that conspecfunction.storeandupdate outputs the intrinsic rewards for the current minibatch, which must be added to the RL agent's reward buffer for this minibatch. This is the purpose of the 3rd line in (3) above. 
