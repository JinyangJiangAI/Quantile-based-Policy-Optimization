import gym
from gym import spaces
import numpy as np


class ToyEnv(gym.Env):
    """
    Toy Env
    """
    def __init__(self, n=10):
        self.n = n
        self.action_space = spaces.Box(-np.inf*np.ones((1,)), np.inf*np.ones((1,)), dtype=np.float32)
        self.observation_space = spaces.Box(np.zeros((n,)), np.ones((n,)), dtype=np.float32)

        self.order = np.arange(n)
        self.step_count = None
        self.std_buf = None

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg

        std = (action[0] - 1)**2 + 0.1
        self.std_buf.append(std)
        reward = np.random.normal(0, std)

        self.step_count += 1
        state = np.zeros((self.n,))
        if self.step_count==self.n:
            return state, reward, True, {}
        else:
            state[self.order[self.step_count]] += 1.
            return state, reward, False, {}

    def reset(self):
        self.step_count = 0
        self.std_buf = []
        np.random.shuffle(self.order)
        state = np.zeros((self.n,))
        state[self.order[self.step_count]] += 1.
        return state

    def close(self):
        return None

    def render(self, mode=None):
        return np.array(self.std_buf)

# env = ToyEnv()
# observation = env.reset()
# while True:
#     print(observation)
#     action = env.action_space.sample()
#     print(action)
#     observation, reward, done, info = env.step(action)
#     print(reward)
#     if done:
#         break



