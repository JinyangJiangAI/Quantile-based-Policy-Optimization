import os
import re
import random
import argparse
import torch
import datetime
import numpy as np

from agents import QRDQN
from envs import InventoryManagementEnv, VecEnv
from gym.wrappers import FrameStack
import multiprocessing


class Options(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--env_name', type=str, default='InventoryManagement')
        parser.add_argument('--algo_name', type=str, default='QRDQN')
        parser.add_argument('--q_alpha', type=float, default=0.1)

        parser.add_argument('--max_episode', type=int, default=500000)
        parser.add_argument('--gamma', type=float, default=0.99)

        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--lr_decay_freq', type=int, default=2e4)
        parser.add_argument('--lr_decay_rate', type=float, default=0.9)

        parser.add_argument('--est_interval', type=int, default=500)
        parser.add_argument('--log_interval', type=int, default=500)

        parser.add_argument('--workers', type=int, default=50)
        parser.add_argument('--epsilon_max', type=float, default=0.500)
        parser.add_argument('--epsilon_min', type=float, default=0.001)
        parser.add_argument('--batch_size', type=int, default=10000)
        parser.add_argument('--upd_step', type=int, default=10)
        parser.add_argument('--mini_batch', type=int, default=500)
        parser.add_argument('--tau', type=float, default=0.999)
        parser.add_argument('--memory_size', type=int, default=200000)
        parser.add_argument('--q_N', type=int, default=16)
        parser.add_argument('--K', type=float, default=0.)
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
    agent = QRDQN(args, env)
    print(args.algo_name + ' running')
    agent.train()


if __name__ == '__main__':
    seed_list = [i for i in range(2)]
    device_list = [0 for i in range(2)]
    zipped_list = list(zip(seed_list, device_list))
    pool = multiprocessing.Pool(processes=2)
    pool.starmap(run, zipped_list)
    pool.close()
    pool.join()

