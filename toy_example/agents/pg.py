import numpy as np
import torch
from torch.distributions import MultivariateNormal
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR

from utils import Memory, Actor


def lr_lambda(k, a, b, c):
    lr = a / ((b + k) ** c)
    return lr


class PG(object):
    """
    Policy Gradient
    """
    def __init__(self, args, env):
        self.device = args.device
        self.path = args.path
        self.log_interval = args.log_interval
        self.est_interval = args.est_interval
        self.q_alpha = args.q_alpha
        self.gamma = args.gamma
        self.max_episode = args.max_episode

        self.env = env
        self.env_name = args.env_name

        state_dim = np.prod(self.env.observation_space.shape)
        action_dim = np.prod(self.env.action_space.shape)
        self.actor = Actor(state_dim, action_dim, args.init_std)

        self.optimizer = Adam(self.actor.parameters(), 1., eps=1e-5)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda k: lr_lambda(k, args.theta_a, args.theta_b, args.theta_c))
        self.MSELoss = torch.nn.MSELoss()

        self.memory = Memory()
        self.writer = SummaryWriter(log_dir=args.path)

    def train(self):
        disc_epi_rewards = []
        for i_episode in range(self.max_episode+1):
            disc_epi_reward, disc_factor, state = 0, 1, self.env.reset()
            while True:
                action = self.choose_action(state)
                state, reward, done, _ = self.env.step(action)
                disc_epi_reward += disc_factor * reward
                disc_factor *= self.gamma
                self.memory.rewards.append(reward)
                self.memory.is_terminals.append(done)
                if done:
                    break
            self.update()
            self.memory.clear()

            disc_epi_rewards.append(disc_epi_reward)
            self.writer.add_scalar('disc_reward/raw_reward', disc_epi_reward, i_episode)

            if i_episode % self.log_interval == 0 and i_episode != 0:
                lb = max(0,len(disc_epi_rewards)-self.est_interval)
                disc_a_reward, disc_q_reward = np.mean(disc_epi_rewards[lb:]), np.percentile(disc_epi_rewards[lb:], self.q_alpha*100)
                error_w = np.mean((self.actor.model[0].weight.data.cpu().numpy().squeeze(0)-np.ones((self.env.n,)))**2)
                self.writer.add_scalar('error/weights', error_w, i_episode)
                self.writer.add_scalar('disc_reward/aver_reward', disc_a_reward, i_episode)
                self.writer.add_scalar('disc_reward/quantile_reward', disc_q_reward, i_episode)
                print(f'Epi:{i_episode:05d} || disc_a_r:{disc_a_reward:.03f} disc_q_r:{disc_q_reward:.03f}')
                print(f'Epi:{i_episode:05d} || model Updated with lr:{self.scheduler.get_last_lr()[0]:.2e}\n')

            self.scheduler.step()

    def choose_action(self, state):
        state = torch.from_numpy(state).float()
        mean = self.actor(state)
        var = torch.diag(torch.exp(2*self.actor.log_std))
        dist = MultivariateNormal(mean, var)
        action = dist.sample()
        self.memory.states.append(state)
        self.memory.actions.append(action)
        return action.detach().data.cpu().numpy()

    def evaluate(self, state, action):
        mean = self.actor(state)
        var = torch.diag(torch.exp(2*self.actor.log_std))
        dist = MultivariateNormal(mean, var)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_logprobs, dist_entropy

    def compute_reward2go(self):
        memory_len = self.memory.get_len()
        reward2go = np.zeros(memory_len, dtype=float)
        pre_r_sum = 0
        for i in range(memory_len - 1, -1, -1):
            if self.memory.is_terminals[i]:
                pre_r_sum = 0
            reward2go[i] = self.memory.rewards[i] + self.gamma * pre_r_sum
            pre_r_sum = reward2go[i]
        reward2go = torch.from_numpy(reward2go).to(self.device).float()
        return reward2go

    def update(self):
        self.actor.to(self.device)

        reward2go = self.compute_reward2go()
        reward2go = (reward2go - reward2go.mean()) / (reward2go.std() + 1e-6)

        old_states = torch.stack(self.memory.states).to(self.device).detach()
        old_actions = torch.stack(self.memory.actions).to(self.device).detach()

        logprobs, dist_entropy = self.evaluate(old_states, old_actions)
        loss = -torch.mean(logprobs * reward2go)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.actor.to(torch.device('cpu'))
