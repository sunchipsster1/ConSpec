


# ConSpec

## Information
This code is built on top of several publically available repositories from different sources.

The basic PPO code was taken from the public repository: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail

The pycolab tasks codebase was taken from the public repository: https://github.com/deepmind/deepmind-research/tree/master/tvt

## Installation
All the requirements to create an environment are in the requirements.txt file



### Launching

To run ConSpec on the 4-key task, run the following ( max reward = 1): 


```
python -u main.py  --algo ppoConSpec  --factorC 1. --use-gae --lr 2e-4 --clip-param 0.08 --value-loss-coef 0.5 --num-processes 16   --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.02 --lrCL 20e-4 --choiceCLparams 0 --seed 80001 --expansion 5000 --pycolab_apple_reward_min 0. --pycolab_apple_reward_max 0. --pycolab_final_reward 1. --factorR 0.2  --head 8
```
