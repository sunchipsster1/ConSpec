import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')


    ####################################
    # Arguments pertaining to ConSpec
    ####################################
    parser.add_argument(
        '--num_prototypes',
        type=int,
        default=8,
        help='')
    parser.add_argument(
        '--lrConSpec', type=float, default=20e-4, help='learning rate (default: 7e-4)')
    parser.add_argument(
        '--intrinsicR_scale',
        type=float,
        default=0.2,
        help='')
    ####################################
    # Arguments pertaining the Pycolab tasks
    ####################################

    parser.add_argument(
        '--pycolab_game',
        default='key_to_door4',
        help='key_to_door4, key_to_door2, key_to_door3, ')
    parser.add_argument(
        '--pycolab_apple_reward_min',
        type=float,
        default=0.,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--pycolab_apple_reward_max',
        type=float,
        default=0.,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--pycolab_final_reward',
        type=float,
        default=1.,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--pycolab_num_apples',
        type=int,
        default=10,
        help='')
    parser.add_argument(
        '--pycolab_fix_apple_reward_in_episode',
        action='store_false',
        default=True,
        help='use generalized advantage estimation')
    parser.add_argument(
        '--pycolab_crop',
        action='store_true',
        default=False,
        help='use generalized advantage estimation')

    ####################################
    # Original arguments pertaining to RL algorithm - from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
    ####################################


    parser.add_argument(
        '--skip',
        type=int,
        default=4,
        help='')

    parser.add_argument(
        '--algo', default='ppo', help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument(
        '--gail',
        action='store_true',
        default=False,
        help='do imitation learning with gail')
    parser.add_argument(
        '--gail-experts-dir',
        default='./gail_experts',
        help='directory that contains expert demonstrations for gail')
    parser.add_argument(
        '--gail-batch-size',
        type=int,
        default=128,
        help='')
    parser.add_argument(
        '--gail-epoch', type=int, default=5, help='gail epochs (default: 5)')
    parser.add_argument(
        '--lr', type=float, default=7e-4, help='learning rate (default: 7e-4)')
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.99,
        help='RMSprop optimizer apha (default: 0.99)')

    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--use-gae',
        action='store_true',
        default=False,
        help='use generalized advantage estimation')
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help='gae lambda parameter (default: 0.95)')
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.02,
        help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='max norm of gradients (default: 0.5)')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument(
        '--num-processes',
        type=int,
        default=16,
        help='how many training CPU processes to use (default: 16)')
    parser.add_argument(
        '--num-steps',
        type=int,
        default=5,
        help='number of forward steps in A2C (default: 5)')
    parser.add_argument(
        '--ppo-epoch',
        type=int,
        default=4,
        help='number of ppo epochs (default: 4)')
    parser.add_argument(
        '--num-mini-batch',
        type=int,
        default=4,
        help='number of batches for ppo (default: 32)')
    parser.add_argument(
        '--clip-param',
        type=float,
        default=0.2,
        help='ppo clip parameter (default: 0.2)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=1,
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=100,
        help='save interval, one save per n updates (default: 100)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=None,
        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument(
        '--num-env-steps',
        type=int,
        default=10e8,
        help='number of environment steps to train (default: 10e6)')
    parser.add_argument(
        '--env-name',
        default='PongNoFrameskip-v4',
        help='environment to train on (default: PongNoFrameskip-v4)')
    parser.add_argument(
        '--log-dir',
        default='/tmp/gym/',
        help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument(
        '--save-dir',
        default='./trained_models/',
        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    parser.add_argument(
        '--use-proper-time-limits',
        action='store_true',
        default=False,
        help='compute returns taking into account time limits')
    parser.add_argument(
        '--recurrent-policy',
        action='store_false',
        default=True,
        help='use a recurrent policy')
    parser.add_argument(
        '--use-linear-lr-decay',
        action='store_true',
        default=False,
        help='use a linear schedule on the learning rate')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args
