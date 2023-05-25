import numpy as np
import torch
from torch.distributions import Categorical
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy

from utils import Memory, ActorDiscrete, Critic


def indicator(x, y:torch.Tensor):
    """Indicator"""
    return torch.where(y <= x, torch.ones_like(y), torch.zeros_like(y))


class QPPO(object):
    """
    Quantile-based Proximal Policy Optimization
    """
    def __init__(self, args, env):
        self.device = args.device
        self.path = args.path
        self.log_interval = args.log_interval
        self.est_interval = args.est_interval
        self.q_alpha = args.q_alpha

        self.clip_eps = args.clip_eps
        self.gamma = args.gamma
        self.max_episode = args.max_episode
        self.upd_interval = args.upd_interval
        self.upd_step = args.upd_step
        self.mini_batch = args.mini_batch
        self.vf_coef = args.vf_coef
        self.ent_coef = args.ent_coef

        self.env = env
        self.env_name = args.env_name
        self.workers = len(self.env.env_list)

        state_dim = list(self.env.observation_space.shape)
        action_dim = self.env.action_space.nvec

        self.actor = ActorDiscrete(state_dim, action_dim).to(self.device)
        self.old_actor = ActorDiscrete(state_dim, action_dim).to(self.device)
        self.old_actor.load_state_dict(self.actor.state_dict())
        state_dim[1] += 1
        self.critic = Critic(state_dim).to(self.device)

        self.optimizer = Adam(list(self.actor.parameters()) + list(self.critic.parameters()), args.lr, eps=1e-5)
        self.scheduler = StepLR(self.optimizer, step_size=args.lr_decay_freq, gamma=args.lr_decay_rate)
        self.MSELoss = torch.nn.MSELoss()

        self.memory = Memory()
        self.writer = SummaryWriter(log_dir=args.path)

        self.T, self.T0 = args.T, args.T0
        q = self.warm_up(self.est_interval)
        self.q_est = torch.autograd.Variable(torch.from_numpy(q)).to(self.device)
        self.q_optimizer = Adam([self.q_est], args.q_lr, eps=1e-5)
        self.q_scheduler = StepLR(self.q_optimizer, step_size=args.lr_decay_freq, gamma=(1 + args.lr_decay_rate) / 2)

    def warm_up(self, max_episode):
        env = deepcopy(self.env)
        disc_epi_rewards = np.zeros((max_episode, self.T-self.T0), dtype=float)
        for i_episode in range(0,max_episode,self.workers):
            disc_epi_reward, disc_factor, state = np.zeros((self.workers,)), 1, env.reset()
            while True:
                state = np.array(state).transpose((0,2,1))
                action = self.choose_action(state)
                state, reward, done, _ = env.step(action)
                disc_epi_reward += disc_factor*reward
                disc_factor *= self.gamma
                if (self.memory.t[-1] >= self.T0) and (self.memory.t[-1] <= self.T-1):
                    disc_epi_rewards[i_episode:i_episode+self.workers, self.memory.t[-1]-self.T0] += disc_epi_reward
                self.memory.is_terminals.append(done)
                if done[0]:
                    if self.memory.t[-1] < self.T-1:
                        disc_epi_rewards[i_episode:i_episode+self.workers, max(0, self.memory.t[-1]-self.T0+1):] += disc_epi_reward
                    break
        q = np.percentile(disc_epi_rewards[:np.arange(0,self.max_episode,self.workers)[-1]], self.q_alpha*100, axis=0)
        print(f'QPPO warm up || n_epi:{max_episode:04d} {self.q_alpha:.2f}-quantile')
        self.memory.clear()
        return q

    def train(self):
        disc_epi_rewards = []
        for i_episode in range(0, self.max_episode, self.workers):
            disc_epi_reward, disc_factor, state = np.zeros((self.workers,)), 1, self.env.reset()
            while True:
                state = np.array(state).transpose((0,2,1))
                action = self.choose_action(state)
                state, reward, done, _ = self.env.step(action)
                disc_epi_reward += disc_factor * reward
                disc_factor *= self.gamma
                self.memory.rewards.append(reward)
                self.memory.is_terminals.append(done)
                if done[0]:
                    break
            disc_epi_rewards.extend(disc_epi_reward)
            for i in range(self.workers):
                i_epi = i_episode + i
                self.writer.add_scalar('disc_reward/raw_reward', disc_epi_reward[i], i_epi)

                if i_epi % self.log_interval == 0 and i_epi != 0:
                    lb = max(0, len(disc_epi_rewards) - self.est_interval)
                    disc_a_reward, disc_q_reward = np.mean(disc_epi_rewards[lb:]), np.percentile(disc_epi_rewards[lb:], self.q_alpha * 100)
                    self.writer.add_scalar('disc_reward/aver_reward', disc_a_reward, i_epi)
                    self.writer.add_scalar('disc_reward/quantile_reward', disc_q_reward, i_epi)
                    print(f'Epi:{i_epi:05d} || disc_a_r:{disc_a_reward:.03f} disc_q_r:{disc_q_reward:.03f}')

            if self.memory.get_len() * self.workers >= self.upd_interval:
                self.update()
                self.memory.clear()
                print(f'Epi:{i_episode:05d} || model Updated with lr:{self.scheduler.get_last_lr()[0]:.2e} q_lr:{self.q_scheduler.get_last_lr()[0]:.2e}')
            for i in range(self.workers):
                self.scheduler.step()
                self.q_scheduler.step()

    def choose_action(self, state):
        t = 0 if self.memory.get_len() == 0 or self.memory.is_terminals[-1][0] == True else self.memory.t[-1] + 1
        state = torch.from_numpy(state).float().to(self.device)
        action_probs = self.old_actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = torch.sum(dist.log_prob(action),dim=-1)
        self.memory.t.append(t)
        self.memory.states.append(state)
        self.memory.actions.append(action)
        self.memory.logprobs.append(action_logprob)
        return action.detach().data.cpu().numpy()

    def evaluate(self, state, action, start_idx, t):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = torch.sum(dist.log_prob(action),dim=-1)
        dist_entropy = torch.sum(dist.entropy(),dim=-1)
        tmp = state[start_idx]
        tmp = torch.hstack((tmp,t*torch.ones((len(start_idx),1,tmp.shape[-1]),device=self.device)))
        state_value = self.critic(tmp)
        return action_logprobs, torch.squeeze(state_value), dist_entropy

    def compute_discounted_reward(self):
        memory_len = self.memory.get_len()
        disc_rewards = np.zeros((self.memory.is_terminals.count(True), self.T-self.T0), dtype=float)
        end_idx = []
        i_episode, disc_factor, pre_r_sum = 0, 1, 0
        for i in range(memory_len):
            pre_r_sum = disc_factor * self.memory.rewards[i] + pre_r_sum
            disc_factor *= self.gamma
            if (self.memory.t[i] >= self.T0) and (self.memory.t[i] <= self.T-1):
                disc_rewards[i_episode, self.memory.t[i] - self.T0] += pre_r_sum
            if self.memory.is_terminals[i]:
                if self.memory.t[i] < self.T - 1:
                    disc_rewards[i_episode, max(0, self.memory.t[i] - self.T0 + 1):] += pre_r_sum
                end_idx.append(i)
                i_episode += 1
                disc_factor, pre_r_sum = 1, 0
        disc_rewards = torch.from_numpy(disc_rewards).to(self.device).float()
        n_epi, start_idx = len(end_idx), np.hstack((0, np.array(end_idx)[:-1] + 1))
        epi_len = np.array(end_idx)-start_idx + 1
        return disc_rewards, start_idx.tolist(), end_idx, n_epi, epi_len.tolist()

    def compute_state_value(self, start_idx):
        state_values = torch.zeros((len(start_idx), self.T-self.T0), device=self.device)
        states = torch.stack(self.memory.states).to(self.device).detach()[start_idx]
        for t in range(self.T0,self.T):
            tmp = torch.hstack((states, t * torch.ones((len(start_idx), 1, states.shape[-1]), device=self.device)))
            value = self.critic(tmp)
            state_values[:,t-self.T0] += torch.squeeze(value)
        return state_values.detach()

    def memory_reshape(self):
        memory = Memory()
        for i in range(self.workers):
            memory.actions.extend([x[i] for x in self.memory.actions])
            memory.states.extend([x[i] for x in self.memory.states])
            memory.logprobs.extend([x[i] for x in self.memory.logprobs])
            memory.rewards.extend([x[i] for x in self.memory.rewards])
            memory.is_terminals.extend([x[i] for x in self.memory.is_terminals])
            memory.values.extend([x[i] for x in self.memory.values])
            memory.t.extend([x for x in self.memory.t])
        self.memory = memory

    def update(self):
        self.memory_reshape()

        disc_rewards, start_idx, end_idx, n_epi, epi_len = self.compute_discounted_reward()
        old_state_values = self.compute_state_value(start_idx)

        old_states = torch.stack(self.memory.states).detach()
        old_actions = torch.stack(self.memory.actions).detach()
        old_logprobs = torch.stack(self.memory.logprobs).detach()

        for _ in range(self.upd_step):
            shuffle_l = np.arange(self.T0, self.T) + 1
            np.random.shuffle(shuffle_l)
            for l in shuffle_l:
                shuffle_epi = np.arange(n_epi)
                np.random.shuffle(shuffle_epi)
                batch_len, batch_epi = 0, []
                for i in shuffle_epi:
                    batch_len += min(epi_len[i],l)
                    batch_epi.append(i)
                    if batch_len >= self.mini_batch:

                        idx, start_idx_, end_idx_ = [], [], []
                        for epi in batch_epi:
                            start_idx_.append(len(idx))
                            idx.extend(list(range(start_idx[epi],min(end_idx[epi]+1,start_idx[epi]+l))))
                            end_idx_.append(len(idx)-1)

                        logprobs, state_values, dist_entropy = self.evaluate(old_states[idx], old_actions[idx], start_idx_, l-1)

                        step_log_ratios = logprobs - old_logprobs[idx]
                        log_ratios = torch.stack([torch.sum(step_log_ratios[s:d+1]) for s,d in zip(start_idx_,end_idx_)])
                        ratios = torch.exp(torch.clamp(log_ratios, -20., 20.))

                        ind = indicator(self.q_est.detach()[l-1-self.T0], disc_rewards[batch_epi,l-1-self.T0])
                        advantages = - ind - old_state_values[batch_epi,l-1-self.T0]

                        surr1 = ratios * advantages
                        surr2 = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
                        actor_loss = - torch.min(surr1, surr2).mean()

                        critic_loss = self.MSELoss(state_values, -ind)
                        dist_entropy = dist_entropy.mean()
                        loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * dist_entropy

                        self.optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                        self.optimizer.step()

                        self.q_optimizer.zero_grad()
                        grad = torch.zeros_like(self.q_est, device=self.device)
                        grad[l-1-self.T0] += -torch.mean(self.q_alpha - ratios*ind, dim=0)
                        self.q_est.grad = grad
                        self.q_optimizer.step()

                        batch_len, batch_epi = 0, []

        self.old_actor.load_state_dict(self.actor.state_dict())