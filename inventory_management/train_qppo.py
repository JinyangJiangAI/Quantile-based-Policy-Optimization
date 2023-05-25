import os
import re
import random
import argparse
import torch
import datetime
import numpy as np

from agents import QPPO
from envs import InventoryManagementEnv, VecEnv
from gym.wrappers import FrameStack
import multiprocessing


class Options(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--env_name', type=str, default='InventoryManagement')
        parser.add_argument('--algo_name', type=str, default='QPPO')
        parser.add_argument('--q_alpha', type=float, default=0.1)

        parser.add_argument('--max_episode', type=int, default=500000)
        parser.add_argument('--gamma', type=float, default=0.99)

        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--lr_decay_freq', type=int, default=2e4)
        parser.add_argument('--lr_decay_rate', type=int, default=0.9)

        parser.add_argument('--est_interval', type=int, default=500)
        parser.add_argument('--log_interval', type=int, default=500)

        parser.add_argument('--lambda_gae_adv', type=float, default=0.95)
        parser.add_argument('--clip_eps', type=float, default=0.2)
        parser.add_argument('--vf_coef', type=float, default=0.5)
        parser.add_argument('--ent_coef', type=float, default=0.00)
        parser.add_argument('--upd_interval', type=int, default=10000)
        parser.add_argument('--upd_step', type=int, default=1)
        parser.add_argument('--mini_batch', type=int, default=450)
        parser.add_argument('--workers', type=int, default=50)
        parser.add_argument('--q_lr', type=float, default=2e-1)
        parser.add_argument('--T', type=int, default=100)
        parser.add_argument('--T0', type=int, default=90)
        self.parser = parser

    def parse(self, seed=0, device='0'):
        args = self.parser.parse_args(args=[])
        args.seed = seed
        args.device = torch.device("cuda:"+device if torch.cuda.is_available() else "cpu")

        current_time = re.sub(r'\D', '', str(datetime.datetime.now())[4:-7])
        args.path = './logs/' + args.env_name + '/' + args.algo_name + '_' + current_time + '_' + str(args.seed)
        os.makedirs(args.path)
        return args


def run(seed, device):
    args = Options().parse(seed=seed, device=str(device))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env = InventoryManagementEnv()
    env_list = [FrameStack(InventoryManagementEnv(seed=100 * j + seed), int(np.max(env.l))) for j in range(args.workers)]
    env = VecEnv(env_list)
    agent = QPPO(args, env)
    print(args.algo_name + ' running')
    agent.train()


if __name__ == '__main__':
    seed_list = [i for i in range(4)]
    device_list = [0 for i in range(4)]
    zipped_list = list(zip(seed_list, device_list))
    pool = multiprocessing.Pool(processes=4)
    pool.starmap(run, zipped_list)
    pool.close()
    pool.join()
