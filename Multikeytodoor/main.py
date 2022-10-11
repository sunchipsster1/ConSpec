import copy
import glob
import os
import time
from collections import deque

import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.argumentstvt import get_args
from a2c_ppo_acktr.modelRL import Policy
from a2c_ppo_acktr.modelConSpec import PolicyCL

from a2c_ppo_acktr.storageConSpec import RolloutStorage
from evaluation import evaluate
from moduleConSpec import moduleCL
import traceback

from tensorflow.contrib import framework as contrib_framework

from time import process_time

import time

import numpy as np
from six.moves import range
from six.moves import zip
import tensorflow.compat.v1 as tf

from tvt import batch_env
from tvt import nest_utils
from tvt.pycolab import env as pycolab_env
import sys
import numpy

def main():
    args = get_args()

    keysUsed = np.zeros((args.head,))
    goodones = np.zeros((args.head,))

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    if args.expansion == 500:
        args.pycolab_game = 'key_to_door5_2keyseasy'
    elif args.expansion == 5000:
        args.pycolab_game = 'key_to_door5_4keyseasy'
    elif args.expansion == 50000:
        args.pycolab_game = 'key_to_door5_3keyseasy'
    elif args.expansion == 24:
        args.pycolab_game = 'key_to_doormany4'



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

    actor_criticCL = PolicyCL(
        obsspace,
        env.num_actions,
        base_kwargs={'recurrent': args.recurrent_policy})  # envs.observation_space.shape,
    actor_criticCL.to(device)

    moduleuse = moduleCL(input_size=512, hidden_size=1010, head=args.head, device=device, args=args)
    moduleuse.to(device)
    keysUsedt = torch.FloatTensor(keysUsed).to(device=device)
    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)


    elif args.algo == 'ppoConSpec':
        agent = algo.PPOConSpec(
            actor_critic, actor_criticCL,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            moduleuse,
            args.choiceCLparams,
            args,
            lrCL=args.lrCL,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              obsspace, env.num_actions,
                              actor_critic.recurrent_hidden_state_size, args.head)  # envs.observation_space.shape

    rollouts.to(device)


    obs, _ = envs.reset()

    obs = (torch.from_numpy(obs)).permute((0, 3, 1, 2)).to(device)
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)


    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes


    for j in range(10000):

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
            # print(step)
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
            bad_masks = masks #torch.FloatTensor([[1.0]])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)
        try:
            with torch.no_grad():
                next_value = actor_critic.get_value(
                    rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                    rollouts.masks[-1]).detach()


            ###############################
            with torch.no_grad():
                # now compute new rewards
                rewardstotal = rollouts.retrieveR()
                episode_rewards.append(rewardstotal.sum(0).mean().cpu().detach().numpy())

                rollouts.addPosNeg(1, device, args)
                rollouts.addPosNeg(0, device, args)

                agent.fictitiousReward(rollouts, keysUsedt, device, j)

            ###############################

            rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                     args.gae_lambda, args.use_proper_time_limits)

            value_loss, action_loss, dist_entropy, keysUsed, goodones = agent.update(rollouts, args.head, keysUsed,
                                                                                     goodones, j)
            keysUsedt = torch.FloatTensor(keysUsed).to(device=device) #for the next fictitius reward

            rollouts.after_update()

            if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
                save_path = os.path.join(args.save_dir, args.algo)
                try:
                    os.makedirs(save_path)
                except OSError:
                    pass

                torch.save([
                    actor_critic,
                    getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
                ], os.path.join(save_path, args.env_name + ".pt"))
            if j % args.log_interval == 0 and len(episode_rewards) > 1:
                total_num_steps = (j + 1) * args.num_processes * args.num_steps
                end = time.time()
                print(
                    "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                        .format(j, total_num_steps,
                                int(total_num_steps / (end - start)),
                                len(episode_rewards), rewardstotal[-10:,:].sum(0).mean().cpu().detach().numpy() ,
                                np.median(episode_rewards), np.min(episode_rewards),
                                np.max(episode_rewards), dist_entropy, value_loss,
                                action_loss))
                LOGFILE = './Results' + str(args.algo)  + 'lr' + str(args.lr) + 'lrCL' + str(args.lrCL) + 'exp'+ str(args.expansion)  + 'factor' + str(args.factorR)+ 'factorC' + str(args.factorC)+ 'finalR' + str(args.pycolab_final_reward) + 'entropy' + str(args.entropy_coef)  + 'seed' + str(args.seed)+ '.txt'
                print(LOGFILE)
                try:
                    printlog1 = f' final costs {total_num_steps} {rewardstotal[-10:,:].sum(0).mean().cpu().detach().numpy()} {episode_rewards[-1]}  {episode_rewards[-1]} {np.mean(episode_rewards)} {np.median(episode_rewards)} {dist_entropy} {value_loss} {action_loss} \n'
                    with open(LOGFILE, 'a') as f:
                        f.write(printlog1)

                except:
                    pass

            if (args.eval_interval is not None and len(episode_rewards) > 1
                    and j % args.eval_interval == 0):  # a
                obs_rms = utils.get_vec_normalize(envs).obs_rms
                evaluate(actor_critic, obs_rms, args.env_name, args.seed,
                         args.num_processes, eval_log_dir, device)


        except:

            pass

if __name__ == "__main__":
    main()
