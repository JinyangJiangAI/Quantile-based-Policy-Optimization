import numpy as np
import torch
from torch.distributions import Categorical
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

from utils import Memory, ActorDiscrete, Critic


class PPO(object):
    """
    Proximal Policy Optimization
    """
    def __init__(self, args, env):
        self.device = args.device
        self.path = args.path
        self.log_interval = args.log_interval
        self.est_interval = args.est_interval
        self.q_alpha = args.q_alpha
        self.clip_eps = args.clip_eps
        self.gamma = args.gamma
        self.lambda_gae_adv = args.lambda_gae_adv
        self.max_episode = args.max_episode
        self.upd_interval = args.upd_interval
        self.upd_step = args.upd_step
        self.mini_batch = args.mini_batch
        self.vf_coef = args.vf_coef
        self.ent_coef = args.ent_coef

        self.env = env
        self.env_name = args.env_name

        state_dim = np.prod(self.env.observation_space.shape)
        action_dim = self.env.action_space.n
        self.actor = ActorDiscrete(state_dim, action_dim, args.emb_dim)
        self.old_actor = ActorDiscrete(state_dim, action_dim, args.emb_dim)
        self.old_actor.load_state_dict(self.actor.state_dict())
        self.critic = Critic(state_dim, args.emb_dim)

        self.optimizer = Adam(list(self.actor.parameters()) + list(self.critic.parameters()), args.lr, eps=1e-5)
        self.scheduler = StepLR(self.optimizer, step_size=args.lr_decay_freq, gamma=args.lr_decay_rate)
        self.MSELoss = torch.nn.MSELoss()

        self.memory = Memory()
        self.writer = SummaryWriter(log_dir=args.path)

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
                    state = state.flatten()
                    last_value = self.critic(torch.from_numpy(state).float()).detach().data.numpy()
                    self.memory.last_values.append(last_value)
                    break
            freq = self.env.render()
            self.writer.add_scalar('acc/total_accuracy', freq[0], i_episode)
            disc_epi_rewards.append(disc_epi_reward)
            self.writer.add_scalar('disc_reward/raw_reward', disc_epi_reward, i_episode)

            if i_episode % self.log_interval == 0 and i_episode != 0:
                lb = max(0, len(disc_epi_rewards) - self.est_interval)
                disc_a_reward, disc_q_reward = np.mean(disc_epi_rewards[lb:]), np.percentile(disc_epi_rewards[lb:], self.q_alpha * 100)
                self.writer.add_scalar('disc_reward/aver_reward', disc_a_reward, i_episode)
                self.writer.add_scalar('disc_reward/quantile_reward', disc_q_reward, i_episode)
                print(f'Epi:{i_episode:05d} || disc_a_r:{disc_a_reward:.03f} disc_q_r:{disc_q_reward:.03f}')

            if self.memory.get_len() >= self.upd_interval:
                self.update()
                self.memory.clear()
                print(f'Epi:{i_episode:05d} || model Updated with lr:{self.scheduler.get_last_lr()[0]:.2e}')
            self.scheduler.step()

    def choose_action(self, state):
        state = torch.from_numpy(state).float()
        action_probs = self.old_actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        self.memory.states.append(state)
        self.memory.actions.append(action)
        self.memory.logprobs.append(action_logprob)
        value = self.critic(state).detach().data.numpy()
        self.memory.values.append(value)
        return action.detach().data.numpy()

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)
        return action_logprobs, torch.squeeze(state_value), dist_entropy

    def compute_reward2go_gae(self):
        memory_len = self.memory.get_len()
        reward2go = np.zeros(memory_len, dtype=float)
        advantage_value = np.zeros(memory_len, dtype=float)
        pre_r_sum = 0
        pre_adv_v = 0
        for i in range(memory_len - 1, -1, -1):
            if self.memory.is_terminals[i]:
                pre_r_sum = 0
                pre_adv_v = self.memory.last_values.pop()
            reward2go[i] = self.memory.rewards[i] + self.gamma * pre_r_sum
            pre_r_sum = reward2go[i]
            advantage_value[i] = self.memory.rewards[i] + self.gamma * pre_adv_v - self.memory.values[i]
            pre_adv_v = self.memory.values[i] + advantage_value[i] * self.lambda_gae_adv
        reward2go = torch.from_numpy(reward2go).to(self.device).float()
        advantage_value = torch.from_numpy(advantage_value).to(self.device).float()
        return reward2go, advantage_value

    def update(self):
        self.actor.to(self.device)
        self.critic.to(self.device)

        reward2go, advantage_value = self.compute_reward2go_gae()
        advantage_value = (advantage_value - advantage_value.mean()) / (advantage_value.std() + 1e-6)

        old_states = torch.stack(self.memory.states).to(self.device).detach()
        old_actions = torch.stack(self.memory.actions).to(self.device).detach()
        old_logprobs = torch.stack(self.memory.logprobs).to(self.device).detach()

        n_data = len(self.memory.states)
        shuffle_idx = np.arange(n_data)
        for _ in range(self.upd_step):
            np.random.shuffle(shuffle_idx)
            for i in range(n_data//self.mini_batch):
                if i == n_data//self.mini_batch - 1:
                    idx = shuffle_idx[self.mini_batch*i: n_data-1]
                else:
                    idx = shuffle_idx[self.mini_batch*i: self.mini_batch*(i+1)]

                logprobs, state_values, dist_entropy = self.evaluate(old_states[idx], old_actions[idx])
                ratios = torch.exp(logprobs - old_logprobs[idx])
                surr1 = ratios * advantage_value[idx]
                surr2 = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * advantage_value[idx]
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss =  self.MSELoss(state_values, reward2go[idx]).mean()
                dist_entropy = dist_entropy.mean()
                loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * dist_entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer.step()

        self.actor.to(torch.device('cpu'))
        self.critic.to(torch.device('cpu'))
        self.old_actor.load_state_dict(self.actor.state_dict())