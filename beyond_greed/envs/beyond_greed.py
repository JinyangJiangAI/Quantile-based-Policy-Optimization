import gym
from gym import spaces
import numpy as np


class BeyondGreedEnv(gym.Env):
    """
    Beyond Greed Env
    """
    def __init__(self, seed=None):
        self.max_step = 10
        self.reward_dist = np.array([[[0., 1.], [0.4, 0.6]],
                                     [[0.5, 1.5], [0.6, 0.4]]])
        # self.reward_dist = np.array([[[0., 1.], [0.3, 0.7]],
        #                              [[0.5, 1.5], [0.7, 0.3]],
        #                              [[0.3, 1.3], [0.4, 0.6]]])

        lb = np.array([0.]*(self.max_step+1), dtype=np.float16)
        ub = np.array([1.]*(self.max_step+1), dtype=np.float16)
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(lb, ub, dtype=np.float16)

        self.state = None
        self.step_count = None
        self.render_buf = None

        self.np_random = None
        self.seed(seed)

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        self.render_buf[action] += 1

        reward = self.np_random.choice(self.reward_dist[action, 0, :], p=self.reward_dist[action, 1, :])
        
        self.step_count += 1
        state = one_hot(self.max_step+1, self.step_count)
        done = bool(self.step_count==self.max_step)
        return state, reward, done, {}

    def reset(self):
        self.step_count = 0
        state = one_hot(self.max_step+1, self.step_count)
        self.render_buf = np.zeros((2,))
        return state

    def render(self, log_dir=None):
        return self.render_buf

    def close(self):
        return None

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    

def one_hot(n, i):
    return np.eye(n)[i]

