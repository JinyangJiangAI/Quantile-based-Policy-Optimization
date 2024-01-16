import gym
from gym import spaces
import numpy as np


class FairLotteryEnv(gym.Env):
    """
    Fair Lottery Env
    """
    def __init__(self):
        self.risk_level = np.array([1,4,9.])
        # self.risk_level = np.array([0.1,0.2,0.3,0.4,0.5])
        self.problem_dim = self.risk_level.shape[0]
        lb = np.array(self.problem_dim*[-np.inf], dtype=np.float32)
        ub = np.array(self.problem_dim*[np.inf], dtype=np.float32)
        self.action_space = spaces.Discrete(self.problem_dim)
        self.observation_space = spaces.Box(lb, ub, dtype=np.float32)
        self.state = None
        self.max_step = 20
        self.step_count = None
        self.render_buf = None
        self.render_buf_short = None

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg

        self.render_buf += np.where(self.risk_level[action] == np.sort(self.risk_level), 1, 0)
        if self.step_count < self.max_step//2:
            self.render_buf_short += np.where(self.risk_level[action] == np.sort(self.risk_level), 1, 0)

        reward = np.random.uniform(-self.risk_level[action],self.risk_level[action])
        np.random.shuffle(self.risk_level)
        state = self.risk_level
        self.step_count += 1

        done = bool(self.step_count==self.max_step)
        return state, reward, done, {}

    def reset(self):
        self.step_count = 0
        np.random.shuffle(self.risk_level)
        state = self.risk_level
        self.render_buf = np.zeros_like(self.risk_level)
        self.render_buf_short = np.zeros_like(self.risk_level)
        return state

    def render(self, log_dir=None):
        freq = self.render_buf / self.max_step
        return freq

    def close(self):
        return None


# env = FairLotteryEnv()
# observation = env.reset()
# while True:
#     observation = np.array(observation)
#     action = np.argmin(observation)
#     # action = env.action_space.sample()
#     observation, reward, done, info = env.step(action)
#     if done:
#         break
# freq = env.render()
# print(freq)


