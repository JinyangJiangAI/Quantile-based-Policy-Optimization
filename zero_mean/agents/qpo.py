import numpy as np
import torch
from torch.distributions import Categorical
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

from utils import Memory, ActorDiscrete


def indicator(x, y:torch.Tensor):
    """Indicator"""
    return torch.where(y <= x, torch.ones_like(y), torch.zeros_like(y))


class QPO(object):
    """
    Quantile-based Policy Gradient
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
        action_dim = self.env.action_space.n
        self.actor = ActorDiscrete(state_dim, action_dim, args.emb_dim)

        self.optimizer = Adam(self.actor.parameters(), args.lr, eps=1e-5)
        self.scheduler = StepLR(self.optimizer, step_size=args.lr_decay_freq, gamma=args.lr_decay_rate)
        self.MSELoss = torch.nn.MSELoss()

        self.memory = Memory()
        self.writer = SummaryWriter(log_dir=args.path)

        q = self.warm_up(5*self.est_interval)
        self.q_est = torch.autograd.Variable(q*torch.ones((1,))).to(self.device)
        self.q_optimizer = Adam([self.q_est], args.q_lr, eps=1e-5)
        self.q_scheduler = StepLR(self.q_optimizer, step_size=args.lr_decay_freq, gamma=(1+args.lr_decay_rate)/2)

    def warm_up(self, max_episode):
        disc_epi_rewards = []
        for _ in range(max_episode):
            disc_epi_reward, disc_factor, state = 0, 1, self.env.reset()
            while True:
                state = state.flatten()
                action = self.choose_action(state)
                state, reward, done, _ = self.env.step(action)
                disc_epi_reward += disc_factor*reward
                disc_factor *= self.gamma
                if done:
                    break
            disc_epi_rewards.append(disc_epi_reward)
        q = np.percentile(disc_epi_rewards, self.q_alpha*100)
        print(f'QPO warm up || n_epi:{max_episode:04d} {self.q_alpha:.2f}-quantile:{q:.3f}')
        self.memory.clear()
        return q

    def train(self):
        disc_epi_rewards = []
        for i_episode in range(self.max_episode+1):
            disc_epi_reward, disc_factor, state = 0, 1, self.env.reset()
            while True:
                state = state.flatten()
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

            freq = self.env.render()
            self.writer.add_scalar('acc/total_accuracy', freq[0], i_episode)
            disc_epi_rewards.append(disc_epi_reward)
            self.writer.add_scalar('disc_reward/raw_reward', disc_epi_reward, i_episode)

            if i_episode % self.log_interval == 0 and i_episode != 0:
                lb = max(0,len(disc_epi_rewards)-self.est_interval)
                disc_a_reward, disc_q_reward = np.mean(disc_epi_rewards[lb:]), np.percentile(disc_epi_rewards[lb:], self.q_alpha*100)
                self.writer.add_scalar('disc_reward/aver_reward', disc_a_reward, i_episode)
                self.writer.add_scalar('disc_reward/quantile_reward', disc_q_reward, i_episode)
                print(f'Epi:{i_episode:05d} || disc_a_r:{disc_a_reward:.03f} disc_q_r:{disc_q_reward:.03f}')
                print(f'Epi:{i_episode:05d} || model Updated with lr:{self.scheduler.get_last_lr()[0]:.2e} '
                      + f'q_lr:{self.q_scheduler.get_last_lr()[0]:.2e}\n')
            self.scheduler.step()
            self.q_scheduler.step()

    def choose_action(self, state):
        state = torch.from_numpy(state).float()
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        self.memory.states.append(state)
        self.memory.actions.append(action)
        return action.detach().data.numpy()

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_logprobs, dist_entropy

    def compute_discounted_epi_reward(self):
        memory_len = self.memory.get_len()
        disc_reward, disc_reward_short = np.zeros(memory_len, dtype=float), []
        pre_r_sum, p1, p2 = 0, 0, 0
        for i in range(memory_len - 1, -1, -1):
            if self.memory.is_terminals[i]:
                if p1 > 0:
                    disc_reward[memory_len-p1: memory_len-p2] += pre_r_sum
                    disc_reward_short.insert(0, pre_r_sum)
                pre_r_sum, p2 = 0, p1
            pre_r_sum = self.memory.rewards[i] + self.gamma * pre_r_sum
            p1 += 1
        disc_reward[memory_len-p1: memory_len-p2] += pre_r_sum
        disc_reward_short.insert(0, pre_r_sum)
        disc_reward = torch.from_numpy(disc_reward).to(self.device).float()
        disc_reward_short = torch.tensor(disc_reward_short).to(self.device).float()
        return disc_reward, disc_reward_short

    def update(self):
        self.actor.to(self.device)
        disc_reward, disc_reward_short = self.compute_discounted_epi_reward()
        ind = indicator(self.q_est.detach(), disc_reward)
        old_states = torch.stack(self.memory.states).to(self.device).detach()
        old_actions = torch.stack(self.memory.actions).to(self.device).detach()
        logprobs, dist_entropy = self.evaluate(old_states, old_actions)
        loss = torch.mean(logprobs * ind)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.q_optimizer.zero_grad()
        self.q_est.grad = -torch.mean(self.q_alpha - indicator(self.q_est, disc_reward_short), dim=0, keepdim=True)
        self.q_optimizer.step()
        self.actor.to(torch.device('cpu'))
