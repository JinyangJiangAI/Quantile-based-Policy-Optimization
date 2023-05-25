import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


class PortfolioManagementEnv(gym.Env):
    """
    Portfolio Management Env
    """
    def __init__(self, seed=0):
        # self.simulator = BlackScholes(num_stocks=3,
        #                               mu=np.array([0.01, 0.08, 0.16]),
        #                               sigma=np.array([[0.01, 0.00, 0.00],
        #                                               [0.00, 0.08, -0.08],
        #                                               [0.00, -0.08, 0.08]]),
        #                               S0=np.array([1., 1., 1.]))
        self.simulator = BlackScholes(num_stocks=5,
                                      mu=np.array([0.01, 0.02, 0.03, 0.04, 0.05]),
                                      sigma=np.array([[0.01, 0.00, 0.00, 0.00, 0.00],
                                                      [0.00, 0.04, -0.055, 0.00, 0.00],
                                                      [0.00, -0.055, 0.09, 0.00, 0.00],
                                                      [0.00, 0.00, 0.00, 0.16, -0.19],
                                                      [0.00, 0.00, 0.00, -0.19, 0.25]]),
                                      S0=np.array([1., 1., 1., 1., 1.]))

        self.n_stocks = self.simulator.num_stocks
        self.max_steps = 100
        self.est_window = 25
        self.dt = 1/100

        self.action_space = spaces.Box(np.array(self.n_stocks*[-np.inf]),
                                       np.array(self.n_stocks*[np.inf]), dtype=float)
        self.observation_space = spaces.Box(-np.inf*np.ones((3*self.n_stocks+ self.n_stocks*(self.n_stocks+1)//2,)),
                                            np.inf*np.ones((3*self.n_stocks+ self.n_stocks*(self.n_stocks+1)//2,)), dtype=float)

        self.transaction_cost = 0.001
        self.init_value = 100.

        self.t = None
        self.price = None
        self.alloc = None
        self.value = None
        self.reward = None
        self.position = None
        self.est_mu = None
        self.est_sigma = None
        self.smooth_coef = 0.0
        tril_mask = [[j<=i for j in range(self.n_stocks)] for i in range(self.n_stocks)]
        self.tril_mask = [item for sublist in tril_mask for item in sublist]

        self.seed(seed)

    def step(self, action: np.ndarray, is_rl=True):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        if is_rl:
            action = np.exp(action) / np.sum(np.exp(action))
        self.alloc[:,self.t+1] += action

        stock_value = self.position[:, self.t] * self.price[:,self.t+self.est_window-1]
        stock_attempt = self.alloc[:,self.t+1] * self.value[self.t]
        delta_position = (stock_attempt - stock_value) / self.price[:,self.t+self.est_window-1]
        delta_position = np.where(delta_position>0, delta_position*(1-self.transaction_cost),delta_position)
        self.position[:, self.t+1] += self.position[:, self.t] + delta_position

        self.price[:,self.t+self.est_window] += self.simulator.step(dt=self.dt)
        self.value[self.t+1] += np.sum(self.position[:, self.t+1]*self.price[:,self.t+self.est_window])
        self.reward[self.t] += self.value[self.t+1] - self.value[self.t]
        reward = self.reward[self.t]

        self.t += 1
        ratio = (self.price[:,self.t+1:self.t+self.est_window] - self.price[:,self.t:self.t+self.est_window-1]) \
                / self.price[:,self.t:self.t+self.est_window-1]
        self.est_mu[:,self.t] += self.smooth_coef*self.est_mu[:,self.t-1] + (1-self.smooth_coef)*np.mean(ratio, axis=1)
        tmp = self.smooth_coef*self.est_sigma[:,self.t-1] + (1-self.smooth_coef)*np.cov(ratio, rowvar=True).flatten()
        self.est_sigma[:,self.t] += tmp

        tmp = tmp[self.tril_mask]
        state = np.concatenate((self.est_mu[:,self.t], tmp, self.position[:,self.t],self.price[:,self.t+self.est_window-1]))

        done = self.t == self.max_steps

        return state, reward, done, {}

    def reset(self):
        self.t = 0
        self.price = np.zeros((self.n_stocks, self.max_steps + self.est_window))
        self.value = np.zeros((self.max_steps + 1,))
        self.reward = np.zeros((self.max_steps,))
        self.alloc = np.zeros((self.n_stocks, self.max_steps + 1))
        self.position = np.zeros((self.n_stocks, self.max_steps + 1))

        self.est_mu = np.zeros((self.n_stocks, self.max_steps + 1))
        self.est_sigma = np.zeros((self.n_stocks**2, self.max_steps + 1))

        self.price[:,0] += self.simulator.reset()
        for i in range(1,self.est_window):
            self.price[:,i] += self.simulator.step(dt=self.dt)
        self.simulator.stock_prices /= (self.price[:,self.est_window-1]/self.simulator.S0)
        self.price[:,:self.est_window] /= (self.price[:,self.est_window-1]/self.simulator.S0[np.newaxis,:]).T # 在t=0处归化
        self.value[0] += self.init_value

        action = self.action_space.sample()
        action = np.exp(action) / np.sum(np.exp(action))
        self.alloc[:,0] += action
        self.position[:, 0] += action * self.init_value / self.price[:, self.est_window - 1]

        ratio = (self.price[:,1:self.est_window]-self.price[:,:self.est_window-1]) / self.price[:,:self.est_window-1]

        self.est_mu[:,self.t] += np.mean(ratio, axis=1)
        self.est_sigma[:,self.t] += np.cov(ratio, rowvar=True).flatten()

        tmp = self.est_sigma[:,self.t][self.tril_mask]
        state = np.concatenate((self.est_mu[:, 0], tmp, self.position[:, 0], self.price[:, self.est_window - 1]))
        return state

    def price_render(self, log_dir=None):
        fig = plt.figure(figsize=(6,2.5))
        ax = fig.add_subplot(111)
        for i in range(self.n_stocks):
            ax.plot(np.arange(-1,self.max_steps),
                    self.price[i,self.est_window-1:], label='stock '+str(i), lw=1)
        ax.set_ylabel('price')
        ax.set_xlim([-1, self.max_steps])
        ax.legend(loc='upper left', fontsize='x-small', markerscale=0.5, ncol=self.n_stocks//2)
        ax.set_xlabel('t')
        plt.tight_layout()
        if log_dir is None:
            plt.show()
        else:
            plt.savefig(log_dir, bbox_inches='tight')

    def action_render(self, log_dir=None):
        fig = plt.figure(figsize=(6,2.5))
        ax = fig.add_subplot(111)
        ax.stackplot(np.arange(-1, self.max_steps), self.alloc, labels=['stock '+str(i) for i in range(self.n_stocks)])
        ax.set_ylabel('allocation')
        ax.set_ylim([0,1])
        ax.set_xlim([-1, self.max_steps])
        ax.legend(loc='upper left', fontsize='x-small', markerscale=0.5, ncol=self.n_stocks//2)
        ax.set_xlabel('t')
        plt.tight_layout()
        if log_dir is None:
            plt.show()
        else:
            plt.savefig(log_dir, bbox_inches='tight')

    def position_render(self, log_dir=None):
        fig = plt.figure(figsize=(6,2.5))
        ax = fig.add_subplot(111)
        for i in range(self.n_stocks):
            ax.plot(np.arange(-1, self.max_steps), self.position[i,:], label='stock ' + str(i), lw=1)
        ax.set_ylabel('position')
        ax.set_xlim([-1, self.max_steps])
        ax.legend(loc='upper left', fontsize='x-small', markerscale=0.5, ncol=self.n_stocks//2)
        ax.set_xlabel('t')
        plt.tight_layout()
        if log_dir is None:
            plt.show()
        else:
            plt.savefig(log_dir, bbox_inches='tight')

    def close(self):
        return None

    def render(self, mode="human"):
        return None


class BlackScholes:
    def __init__(self, num_stocks, mu, sigma, S0):
        self.num_stocks = num_stocks
        self.mu = mu
        self.sigma = sigma
        self.S0 = S0

    def reset(self):
        self.stock_prices = np.copy(self.S0)
        return self.stock_prices

    def step(self, dt):
        dS = self.stock_prices * np.random.multivariate_normal(self.mu*dt, self.sigma*dt)
        self.stock_prices += dS
        return self.stock_prices


def markowitz(mu, sigma):
    def weight_sum_constraint(weights):
        return np.sum(weights) - 1.0

    def obj_func(weights):
        var = np.dot(weights.T, np.dot(sigma, weights))
        mean = np.dot(weights.T, mu)
        return - mean + 1.28*np.sqrt(var)

    init_weights = np.ones(mu.shape[0]) / mu.shape[0]
    bounds = [(0.0, 1.0) for _ in range(mu.shape[0])]
    constraints = [{'type': 'eq', 'fun': weight_sum_constraint}]
    result = minimize(obj_func, init_weights, method='SLSQP', bounds=bounds, constraints=constraints, tol=1e-10)
    return result.x


# env = PortfolioManagementEnv()
# observation = env.reset()
# while True:
#     observation = np.array(observation)
#     # action = env.action_space.sample()
#     # _, reward, done, _ = env.step(action)
#     action = markowitz(env.simulator.mu * env.dt, env.simulator.sigma * env.dt)
#     _, reward, done, _ = env.step(action, is_rl=False)
#     if done:
#         break
# env.render()
# env.price_render()
# env.action_render()
# env.position_render()




