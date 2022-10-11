import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.modelRL import Policy
from a2c_ppo_acktr.modelConspec import PolicyCL

from Utils.utils import plot_learning_curve
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
import numpy as np

from a2c_ppo_acktr.storage  import RolloutStorage
from evaluation import evaluate
from moduleConspec import moduleCL
import traceback


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
    #chmod -R 755 /home/chen/PycharmProjects/ProjTreeSimplePink2extrahardLinux4590_338084842miss/GridWorld.x86_64
    envU = UnityEnvironment(r"[directory...]/ProjTreeSimplePink2extrahardLinux4590_338084842miss/GridWorld.x86_64",seed=2,worker_id=3,no_graphics=False)
    envs = UnityToGymWrapper(envU, allow_multiple_obs=True)
    obsspace = (3, 84, 84)
    
    actor_critic = Policy(
        obsspace,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    actor_criticCL = PolicyCL(
        obsspace,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_criticCL.to(device)
    save_path = args.save_dir

    moduleuse = moduleCL(input_size=256, hidden_size=100, head=args.head, device=device, args=args)
    moduleuse.to(device)
    
    keysUsedt = torch.FloatTensor(keysUsed).to(device=device)

    if args.algo == 'Conspec':
        agent = algo.PPOConspec(
            actor_critic,actor_criticCL,
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
                              obsspace, envs.action_space,
                              actor_critic.recurrent_hidden_state_size, args.head)  # envs.observation_space.shape

    minibsize = args.num_processes
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes


    for j in range(1002):
        
        for kkk in range(minibsize):
            obs = envs.reset()
            obs = torch.from_numpy(obs[0]).permute((2,0,1)).to(device)
            rollouts.obs[0,kkk].copy_(obs)
            donetotal = 0
            done = 0
            maskstotal = torch.ones((args.num_processes, 1))  # .to(device)

            if args.use_linear_lr_decay:
                # decrease learning rate linearly
                utils.update_linear_schedule(
                    agent.optimizer, j, num_updates,
                    agent.optimizer.lr if args.algo == "acktr" else args.lr)

            for step in range(args.num_steps):

                if done > 0:
                    if donetotal < 2:
                        donetotal += 1
                        obs = envs.reset()
                        obs = torch.from_numpy(obs[0]).permute((2, 0, 1)).to(device)*255.
                        rollouts.obs[step, kkk].copy_(obs)
                    elif donetotal == 2:
                        donetotal += 1

                # Sample actions
                with torch.no_grad():
                    value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        rollouts.obs[step,kkk].unsqueeze(0).tile((minibsize,1,1,1)), rollouts.recurrent_hidden_states[step,kkk].reshape(-1,1).tile((minibsize,1)),
                        rollouts.masks[step,kkk].reshape(-1,1).tile((minibsize,1)))
                
                if donetotal < 2.5:
                    obs, reward, done, infos = envs.step(action[0].detach().cpu().numpy())
                    obs = torch.from_numpy(obs[0]).unsqueeze(0).permute((0, 3, 1, 2)).to(device)*255.
                    
                    reward = np.floor(reward / 5.) / 10. #normalize rewards since they were arbitrarily set at 50
                    reward = torch.from_numpy((reward).reshape(-1, 1)).reshape(-1, 1)  
                    done = np.ones((1,)) * done  
                    reward = reward * maskstotal  
                    maskstotalobs = torch.tile(
                        torch.unsqueeze(torch.unsqueeze((maskstotal), -1), -1),
                        (1, obs.shape[1], obs.shape[2], obs.shape[3]))
                    obs = obs * maskstotalobs.to(device)
                    masks = torch.FloatTensor(
                        [[1.0] if done_ else [1.0] for done_ in done])
                    maskstotal = masks
                    bad_masks = maskstotal
                else:
                    obs = obs * 0.
                    done = 0


                rollouts.insert(obs[0], recurrent_hidden_states[0], action[0],
                                action_log_prob[0], value[0], reward[0], masks[0], bad_masks[0], kkk)
        try:
            with torch.no_grad():
                next_value = actor_critic.get_value(
                    rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                    rollouts.masks[-1]).detach()

            ###############################
            with torch.no_grad():
                rewardstotal = rollouts.retrieveR()
                episode_rewards.append(rewardstotal[-20:].sum(0).mean().cpu().detach().numpy())
                rollouts.addPosNeg(1, device, args)
                rollouts.addPosNeg(0, device, args)
                agent.fictitiousReward(rollouts, keysUsedt, device, j)

            ###############################

            rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                     args.gae_lambda, args.use_proper_time_limits)

            value_loss, action_loss, dist_entropy, keysUsed, goodones = agent.update(rollouts, args.head, keysUsed,
                                                                                     goodones, j)
            keysUsedt = torch.FloatTensor(keysUsed).to(device=device) #for the next fictitius reward

            if j % args.log_interval == 0 and len(episode_rewards) > 1:
                total_num_steps = (j + 1) * args.num_processes * args.num_steps
                end = time.time()
                print(
                    "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                        .format(j, total_num_steps,
                                int(total_num_steps / (end - start)),
                                len(episode_rewards), np.mean(episode_rewards),
                                np.median(episode_rewards), np.min(episode_rewards),
                                np.max(episode_rewards), dist_entropy, value_loss,
                                action_loss))
                LOGFILE = './Results' + str(args.algo) + 'HEAD' + str(args.head)+ 'choiceCLparam' + str(
                    args.choiceCLparams) + 'lrCL' + str(args.lrCL) + 'seed' + str(args.seed)+ 'factor' + str(args.factorR) + '.txt'
                print(LOGFILE)
                printlog1 = f' results {total_num_steps} {episode_rewards[-1]} \n'
                with open(LOGFILE, 'a') as f:
                    f.write(printlog1)

            if (args.eval_interval is not None and len(episode_rewards) > 1
                    and j % args.eval_interval == 0):  # a
                obs_rms = utils.get_vec_normalize(envs).obs_rms
                evaluate(actor_critic, obs_rms, args.env_name, args.seed,
                         args.num_processes, eval_log_dir, device)

            
            # '''
            if (j > 700) and (j % 50 ==0):  # or (j > 998): # and (j > 500):
                torch.save(actor_critic,
                           os.path.join(save_path,
                                        "model" + str(args.algo) + 'HEAD' + str(args.head)+ 'factor' + str(args.factorR) + str(
                                            args.seed) + 'seed' + str(j) + ".pt"))
                torch.save(actor_criticCL,
                           os.path.join(save_path,
                                        "modelCL" + str(args.algo) + 'HEAD' + str(args.head)+ 'factor' + str(args.factorR) + str(
                                            args.seed) + 'seed' + str(j) + ".pt"))
                torch.save(moduleuse,
                           os.path.join(save_path, "moduleuse" + 'HEAD' + str(args.head)+ str(args.algo) + 'factor' + str(
                               args.factorR) + str(
                               args.seed) + 'seed' + str(j) + ".pt"))

                

        except:
            LOGDIR = './'
            LOGFILE = LOGDIR + 'traceback.txt'
            printlog = f'It: {traceback.format_exc()}  \n'
            with open(LOGFILE, 'a') as f:
                f.write(printlog)



if __name__ == "__main__":
    main()
