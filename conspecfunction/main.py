import copy
import glob
import os
import time
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import numpy
import wandb
from misc import Logger, makedir

########packages to runthe pycolab game
from tvt import batch_env
from tvt import nest_utils
from tvt.pycolab import env as pycolab_env

########packages to run the underlying RL algorithm, mostly identical to https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.modelRL import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate
from tensorflow.contrib import framework as contrib_framework
import numpy as np
from six.moves import range
from six.moves import zip
import tensorflow.compat.v1 as tf



'''
The code below is nearly identical to https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail 
With only minor additions of ConSpec-specific operations that have been commented ########
'''

def main():
    args = get_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    wandb_project_name = 'conspec'
    wandb.init(name='sunchipsster', project=wandb_project_name)
    proj_name = 'Results' + str(args.pycolab_game) + str(args.seed)
    logger = Logger(
        exp_name=proj_name,
        save_dir='/home/mila/c/chen.sun/scratch/conspec',
        print_every=1,
        save_every=10000,
        total_step=10000,
        print_to_stdout=True,
        wandb_project_name=wandb_project_name,
        wandb_tags=[],
        wandb_config=args,
    )

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    env_builder = pycolab_env.PycolabEnvironment
    env_kwargs = {
        'game': args.pycolab_game,
        'num_apples': args.pycolab_num_apples,
        'apple_reward': [args.pycolab_apple_reward_min,
                         args.pycolab_apple_reward_max],
        'fix_apple_reward_in_episode': args.pycolab_fix_apple_reward_in_episode,
        'final_reward': args.pycolab_final_reward,
        'crop': args.pycolab_crop
    }
    env = batch_env.BatchEnv(args.num_processes, env_builder, **env_kwargs)
    ep_length = env.episode_length
    args.num_steps = ep_length

    envs = env
    obsspace = (3,5,5) #env.observation_shape
    actor_critic = Policy(
        obsspace,
        env.num_actions,
        base_kwargs={'recurrent': args.recurrent_policy})  # envs.observation_space.shape,
    actor_critic.to(device)

    ###################################
    ##decide on which underlying RL agent to use - ppo or a2c. But any other RL agent of choice should also work
    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)

    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)

    ###############CONSPEC FUNCTIONS##############################
    '''
    Here, the main ConSpec class is loaded. All the relevant ConSpec functions and objects are contained in this class.
    '''
    from Conspec.ConSpec import ConSpec
    conspecfunction = ConSpec(args,   obsspace,  env.num_actions,  device)
    ##############################################################

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              obsspace, env.num_actions,
                              actor_critic.recurrent_hidden_state_size, args.num_prototypes)  # envs.observation_space.shape
    rollouts.to(device)

    obs, _ = envs.reset()
    obs = (torch.from_numpy(obs)).permute((0, 3, 1, 2)).to(device)
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes

    logger.start()
    for j in range(10000):
        logger.step()
        obs, _ = envs.reset()
        obs = (torch.from_numpy(obs)).permute((0, 3, 1, 2)).to(device)
        rollouts.obs[0].copy_(obs)
        donetotal = np.zeros((args.num_processes,))  # .to(device)
        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            obs, reward = envs.step(action)
            obs = torch.from_numpy(obs).permute((0, 3, 1, 2)).to(device)
            reward = torch.from_numpy(reward).reshape(-1, 1)
            done = donetotal
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = masks
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()
            # now compute new rewards
            rewardstotal = rollouts.retrieveR()
            episode_rewards.append(rewardstotal.sum(0).mean().cpu().detach().numpy())

        ###############CONSPEC FUNCTIONS##############################
        '''
        the purpose here is to: 
        1. retrieve the current minibatch of trajectory (including its observations, rewards, hidden states, actions, masks)
        2. "do everything" that ConSpec needs to do internally for training, and output the intrinsic + extrinsic reward for the current minibatch of trajectories
        3. store this total reward in the memory buffer 
        '''
        obstotal, rewardtotal, recurrent_hidden_statestotal, actiontotal,  maskstotal  = rollouts.release()
        reward_intrinsic_extrinsic  = conspecfunction.do_everything(obstotal, recurrent_hidden_statestotal, actiontotal, rewardtotal, maskstotal)
        rollouts.storereward(reward_intrinsic_extrinsic)
        ##############################################################

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy= agent.update(rollouts)
        rollouts.after_update()

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            logger.meter("results", "R", rewardstotal[-10:,:].sum(0).mean().cpu().detach().numpy())
            logger.meter("results", "dist_entropy", dist_entropy)
            logger.meter("results", "value_loss", value_loss)
            logger.meter("results", "action_loss", action_loss)
    logger.finish()
if __name__ == "__main__":
    main()
