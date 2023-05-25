import os, re
import argparse
import torch
import datetime
import numpy as np
import random
import multiprocessing

from agents import PPO, PG, QPO, QPPO, SPSA
from envs import ZeroMeanEnv


class Options(object):
    def __init__(self, algo_name):
        lr = 1e-3
        q_lr = 1e-2

        parser = argparse.ArgumentParser()
        parser.add_argument('--env_name', type=str, default='ZeroMean')
        parser.add_argument('--algo_name', type=str, default=algo_name)
        parser.add_argument('--q_alpha', type=float, default=0.25)
        parser.add_argument('--est_interval', type=int, default=100)
        parser.add_argument('--log_interval', type=int, default=100)
        parser.add_argument('--max_episode', type=int, default=50000)
        parser.add_argument('--emb_dim', type=list, default=[8,8])
        parser.add_argument('--gamma', type=float, default=0.99)
        parser.add_argument('--lr_decay_freq', type=int, default=2.5e3)
        parser.add_argument('--lr_decay_rate', type=int, default=0.8)

        args = parser.parse_args(args=[])
        if args.algo_name == 'QPO':
            parser.add_argument('--q_lr', type=float, default=q_lr)
            parser.add_argument('--lr', type=float, default=lr)
        if args.algo_name == 'QPPO':
            parser.add_argument('--q_lr', type=float, default=q_lr)
            parser.add_argument('--lr', type=float, default=lr)
            parser.add_argument('--lambda_gae_adv', type=float, default=0.95)
            parser.add_argument('--clip_eps', type=float, default=0.2)
            parser.add_argument('--vf_coef', type=float, default=0.5)
            parser.add_argument('--ent_coef', type=float, default=0.00)
            parser.add_argument('--upd_interval', type=int, default=2000)
            parser.add_argument('--upd_step', type=int, default=5)
            parser.add_argument('--mini_batch', type=int, default=200)
            parser.add_argument('--T', type=int, default=20)
            parser.add_argument('--T0', type=int, default=15)
        if args.algo_name == 'SPSA':
            parser.add_argument('--lr', type=float, default=50*lr) # otherwise SPSA will be too slow
            parser.add_argument('--spsa_batch', type=int, default=5)
            parser.add_argument('--perturb_c', type=float, default=1.9)
            parser.add_argument('--perturb_gamma', type=float, default=1/6)
        if args.algo_name == 'PG':
            parser.add_argument('--lr', type=float, default=lr)
        if args.algo_name == 'PPO':
            parser.add_argument('--lr', type=float, default=lr)
            parser.add_argument('--lambda_gae_adv', type=float, default=0.95)
            parser.add_argument('--clip_eps', type=float, default=0.2)
            parser.add_argument('--vf_coef', type=float, default=0.5)
            parser.add_argument('--ent_coef', type=float, default=0.00)
            parser.add_argument('--upd_interval', type=int, default=2000)
            parser.add_argument('--upd_step', type=int, default=10)
            parser.add_argument('--mini_batch', type=int, default=200)
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

    env = ZeroMeanEnv()

    if args.algo_name == 'PPO':
        agent = PPO(args, env)
    elif args.algo_name == 'QPPO':
        agent = QPPO(args, env)
    elif args.algo_name == 'PG':
        agent = PG(args, env)
    elif args.algo_name == 'QPO':
        agent = QPO(args, env)
    else:
        agent = SPSA(args, env)
    print(args.algo_name + ' running')
    agent.train()


if __name__ == '__main__':
    n = 3
    algos = ['SPSA'] * n
    seeds = [i for i in range(n)]
    devices = [0 for i in range(n)]
    zipped_list = list(zip(algos, seeds, devices))
    pool = multiprocessing.Pool(processes=3)
    pool.starmap(run, zipped_list)
    pool.close()
    pool.join()



