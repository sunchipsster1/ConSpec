


# ConSpec

## Information
This code is built on top of several publically available repositories from different sources.

The basic PPO code was taken from the public repository: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail

The pycolab tasks codebase was taken from the public repository: https://github.com/deepmind/deepmind-research/tree/master/tvt

## Installation
All the requirements to create an environment are in the requirements.txt file



### Launching

To see this function in action with an underlying PPO agent, run: 
```
python -u mainTVTofficial.py  --algo ppo  --factorC 1.0 --use-gae --lr 2e-4 --clip-param 0.08 --value-loss-coef 0.5 --num-processes 16  --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.02 --lrCL 20e-4 --choiceCLparams 0 --seed $seed --expansion 5000 --pycolab_apple_reward_min 0. --pycolab_apple_reward_max 0. --pycolab_final_reward 1. --factorR 0.2  --head 8
```

This function is meant to be autonomous and can be added to any RL agent of choice. 
To use this function, 3 steps need to be done:
1. Import the following:
```
from Conspec.moduleConSpec import moduleCL

from Conspec.conspecfunction import conspecfunctionx
```
2. In your script, define the following: 
```
moduleuse = moduleCL(...)

moduleuse.to(device)

conspecfunction = conspecfunctionx(...)
```
3. To train ConSpec using these functions, run:
```
...  = rollouts.release()

rewardtotalintrisic  = conspecfunction.storeandupdate(... ) 

rollouts.storeintrinsic(rewardtotalintrisic)
```
Note that conspecfunction.storeandupdate outputs the intrinsic rewards for the current minibatch, which must be added to the RL agent's reward buffer for this minibatch. 
This is the purpose of the 3rd line in (3) above. 
