import numpy as np
from .portfolio_management import PortfolioManagementEnv
from multiprocessing import Pool, Queue


class VecEnv(object):
    def __init__(self, env_list):
        #env_list = [PortfolioManagementEnv().seed(i) for i in range(10)]
        self.env_list = env_list
        self.action_space = self.env_list[0].action_space
        self.observation_space = self.env_list[0].observation_space

    def step(self, actions):
        results = [env.step(action) for env, action in zip(self.env_list, actions)]
        obs, rewards, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rewards), np.stack(dones), infos
         
    def reset(self):
        obs = [env.reset() for env in self.env_list]
        return np.stack(obs)

    def close(self):
        pass

    def render(self, log_dir=None, mode="human"):
        self.env_list[0].render(log_dir, mode)



