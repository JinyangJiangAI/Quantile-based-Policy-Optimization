import os, re
import argparse
import torch
import datetime
import numpy as np
import random

from agents import QRDQN, QPO, QPPO
from envs import BeyondGreedEnv
import multiprocessing


class Options(object):
    def __init__(self, algo_name):
        parser = argparse.ArgumentParser()
        parser.add_argument('--env_name', type=str, default='BeyondGreed')
        parser.add_argument('--algo_name', type=str, default=algo_name)
        parser.add_argument('--q_alpha', type=float, default=0.5)
        parser.add_argument('--est_interval', type=int, default=200)
        parser.add_argument('--log_interval', type=int, default=100)
        parser.add_argument('--max_episode', type=int, default=50000)
        parser.add_argument('--emb_dim', type=list, default=[32, 32])
        parser.add_argument('--gamma', type=float, default=0.99)
        parser.add_argument('--lr_decay_freq', type=int, default=5e3)
        parser.add_argument('--lr_decay_rate', type=int, default=0.8)
        parser.add_argument('--lr', type=float, default=5e-4)

        parser.add_argument('--epsilon_max', type=float, default=0.500)
        parser.add_argument('--epsilon_min', type=float, default=0.001)
        parser.add_argument('--batch_size', type=int, default=2000)
        parser.add_argument('--tau', type=float, default=0.999)
        parser.add_argument('--memory_size', type=int, default=200000)
        parser.add_argument('--q_N', type=int, default=32)
        parser.add_argument('--K', type=float, default=0.)

        parser.add_argument('--q_lr', type=float, default=1e-2)

        parser.add_argument('--clip_eps', type=float, default=0.2)
        parser.add_argument('--vf_coef', type=float, default=0.5)
        parser.add_argument('--ent_coef', type=float, default=0.00)
        parser.add_argument('--upd_interval', type=int, default=500)
        parser.add_argument('--upd_step', type=int, default=10)
        parser.add_argument('--mini_batch', type=int, default=50)
        parser.add_argument('--T', type=int, default=10)
        parser.add_argument('--T0', type=int, default=9)
        self.parser = parser

    def parse(self, seed=0, device='0'):
        args = self.parser.parse_args(args=[])
        args.seed = seed
        args.device = torch.device("cuda:" + device if torch.cuda.is_available() else "cpu")

        current_time = re.sub(r'\D', '', str(datetime.datetime.now())[4:-7])
        args.path = './logs/' + args.env_name + '/' + args.algo_name + '_' + current_time + '_' + str(args.seed)
        os.makedirs(args.path)
        return args

def run(algo_name, seed, device):
    args = Options(algo_name).parse(seed, str(device))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = BeyondGreedEnv(seed=args.seed)
    
    if args.algo_name == 'QPO':
        agent = QPO(args, env)
    elif args.algo_name == 'QRDQN':
        agent = QRDQN(args, env)
    elif args.algo_name == 'QPPO':
        agent = QPPO(args, env)
    print(args.algo_name + ' running')
    agent.train()

if __name__ == '__main__':
    n = 1
    algos = ['QPO']*n + ['QPPO']*n + ['QRDQN']*n
    seeds = [i for i in range(n)] + [i for i in range(n)] + [i for i in range(n)]
    devices = [0 for i in range(n)] + [0 for i in range(n)] + [0 for i in range(n)]
    zipped_list = list(zip(algos, seeds, devices))
    pool = multiprocessing.Pool(processes=3*n)
    pool.starmap(run, zipped_list)
    pool.close()
    pool.join()




