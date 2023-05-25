import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


class InventoryManagementEnv(gym.Env):
    """
    Inventory Management Env
    """
    def __init__(self, seed=0):
        self.T = 100                            # 50
        self.l = np.array([2, 3, 5])            # np.array([3])
        self.c = np.array([0.125, 0.1, 0.075])  # np.array([0.1])
        self.h = np.array([0.20, 0.15, 0.10])   # np.array([0.15])
        self.p = np.array([2.0, 1.5, 1.0, 0.5]) # np.array([2,1.5])
        self.init_I = np.array([10, 10, 10])    # np.array([10])
        self.init_S = np.array([10, 10, 10])    # np.array([10])

        self.n_stage = np.size(self.l)
        observation_dim = 4 * self.n_stage
        self.action_space = spaces.MultiDiscrete([21]*self.n_stage)
        self.observation_space = spaces.Box(np.array(observation_dim*[-np.inf], dtype=float),
                                            np.array(observation_dim*[np.inf], dtype=float), dtype=float)

        self.I, self.U, self.S, self.q, self.t, self.R = None, None, None, None, None, None
        # self.demand_sim = Merton()
        self.seed(seed)

    def step(self, action: np.ndarray):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        self.t += 1

        d = [np.random.choice(7) + (self.t+6) % 15]
        # d = self.demand_sim.step()
        # d = [np.random.choice(21)]

        self.q[self.t] += np.concatenate((d, action))
        arrive = [self.S[np.maximum(self.t-self.l[n], 0), n+1] for n in range(self.n_stage)]
        diff = self.q[self.t, :-1] - self.I[self.t - 1] - arrive
        self.S[self.t, :-1] += self.q[self.t, :-1] - np.maximum(diff, 0)
        self.S[self.t,-1] += self.q[self.t,-1]
        self.I[self.t] += self.I[self.t-1] + arrive - self.S[self.t,:-1]
        self.U[self.t] += self.q[self.t, :-1] - self.S[self.t, :-1]

        done = self.t == self.T
        self.R[self.t,:-1] += self.p[:-1] * self.S[self.t,:-1] - self.p[1:] * self.S[self.t,1:] \
                              - self.c * self.U[self.t] - self.h * self.I[self.t]
        self.R[self.t,-1] += np.sum(self.R[self.t,:-1])
        reward = self.R[self.t,-1]
        state = np.hstack((self.I[self.t], self.U[self.t], self.S[self.t,1:], self.q[self.t,:-1]))

        return state, reward, done, {}

    def reset(self):
        self.t = 0
        self.I = np.zeros((self.T+1, self.n_stage))
        self.U = np.zeros((self.T+1, self.n_stage))
        self.S = np.zeros((self.T+1, self.n_stage+1))
        self.q = np.zeros((self.T+1, self.n_stage+1))
        self.R = np.zeros((self.T+1, self.n_stage+1))
        self.I[self.t] += self.init_I
        self.S[self.t,1:] += self.init_S
        # self.demand_sim.reset()
        state = np.hstack((self.I[self.t], self.U[self.t], self.S[self.t,1:], self.q[self.t,:-1]))
        return state

    def render(self, mode="human", log_dir=None):
        fig = plt.figure(figsize=(3*self.n_stage,6))
        colors = list(mcolors.TABLEAU_COLORS.keys())
        for i in range(self.n_stage):
            ax = fig.add_subplot(3,self.n_stage,i+1)
            ax.plot(range(1, self.T + 1), self.q[1:, i], label=r'Echelon ' + str(i - 1) + ' Demand', color='gray', linestyle='--', linewidth=0.8)
            ax.plot(range(1, self.T + 1), self.q[1:, i+1], label=r'Echelon ' + str(i) + ' Order', color=colors[i], linewidth=0.8)
            ax.legend(loc='upper left', fontsize='x-small', markerscale=0.5)
            if i == 0:
                ax.set_ylabel('Demand & Order')
            ax.set_ylim([-1,21])

        for i in range(self.n_stage):
            ax = fig.add_subplot(3,self.n_stage,i+1+self.n_stage)
            ax.plot(range(1, self.T + 1), self.q[1:, i], label=r'Echelon ' + str(i - 1) + ' Demand', color='gray', linestyle='--', linewidth=0.8)
            ax.plot(range(1, self.T + 1), self.I[1:, i], label=r'Echelon ' + str(i) + ' Inventory', color=colors[i], linewidth=0.8)
            ax.plot(range(1, self.T + 1), self.U[1:, i], label=r'Echelon ' + str(i) + ' Shortage', color='black', linewidth=0.8)
            ax.legend(loc='upper left', fontsize='x-small', markerscale=0.5)
            if i == 0:
                ax.set_ylabel('Demand &\nInventory/Shortage')
            ax.set_ylim([-1, 21])

        for i in range(self.n_stage):
            ax = fig.add_subplot(3,self.n_stage,i+1+2*self.n_stage)
            ax.plot(range(1, self.T + 1), self.R[1:, i], label=r'Echelon ' + str(i) + ' Profit', color=colors[i], linewidth=0.8)
            ax.legend(loc='upper left', fontsize='x-small', markerscale=0.5)
            if i == 0:
                ax.set_ylabel('Profit')
            ax.set_xlabel('t')

        plt.tight_layout()
        if log_dir is None:
            plt.show()
        else:
            plt.savefig(log_dir, bbox_inches='tight')

    def close(self):
        return None


class Merton(object):
    def __init__(self):
        self.mu, self.sigma, self.a, self.b, self.lamda, self.U = 5e-5, 1e-2, 0, 0.01, 15, 10
        self.J = None

    def reset(self):
        self.J = 0.

    def step(self):
        Z = np.random.normal(size=(1,))
        Z_ = np.random.normal(size=(1,))
        N = np.random.poisson(lam=self.lamda)
        delta_J = (self.mu - 0.5 * self.sigma ** 2) + self.sigma * Z + self.a * N + self.b * np.sqrt(N) * Z_
        self.J += delta_J
        return np.floor(self.U * np.exp(self.J))


# from gym.wrappers import FrameStack
# env = InventoryManagementEnv()
# env = FrameStack(env, int(np.max(env.l)))
#
# observation = env.reset()
# while True:
#     observation = np.array(observation)
#     action = env.action_space.sample()
#     _, reward, done, info = env.step(action)
#     if done:
#         break
# env.render()

