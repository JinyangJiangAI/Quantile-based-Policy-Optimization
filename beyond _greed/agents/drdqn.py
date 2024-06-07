import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

from models import QNet
from utils import QMemory


class QRDQN(object):
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

        self.state_dim = np.prod(self.env.observation_space.shape)
        try:
            self.action_dim = self.env.action_space.n
        except AttributeError:
            print('Do not support this kind of action space!')
        print(f'state_dim:{self.state_dim}, action_dim:{self.action_dim}')

        self.batch_size = args.batch_size
        self.memory_size = args.memory_size
        self.epsilon_max = args.epsilon_max
        self.epsilon = args.epsilon_max
        self.epsilon_min = args.epsilon_min
        self.tau = args.tau
        self.MSELoss = torch.nn.MSELoss()
        self.memory = QMemory(self.memory_size)
        self.writer = SummaryWriter(log_dir=args.path)
    
        self.q_N = args.q_N
        self.K = args.K
        self.current_net = QNet(self.state_dim, self.action_dim*self.q_N, args.emb_dim)
        self.target_net = QNet(self.state_dim, self.action_dim*self.q_N, args.emb_dim).to(self.device)
        self.optimizer = Adam(self.current_net.parameters(), args.lr)
        self.scheduler = StepLR(self.optimizer, step_size=args.lr_decay_freq, gamma=args.lr_decay_rate)

        self.q_taus = np.array([(2*i - 1)/(2*self.q_N) for i in range(1, self.q_N+1)])
        self.inter_idx = np.sum(self.q_taus < self.q_alpha) # for linear interpolation q_taus[i-1] < q_alpha < q_taus[i]         
        self.slope = (self.q_alpha-self.q_taus[self.inter_idx-1]) / (self.q_taus[self.inter_idx]-self.q_taus[self.inter_idx-1])

    def train(self):
        epi_rewards = []
        disc_epi_rewards = []
        for i_episode in range(self.max_episode + 1):
            epi_reward, disc_epi_reward, disc_factor, state = 0, 0, 1, self.env.reset()
            while True:
                state = state.flatten()
                action = self.choose_action(state)
                state_, reward, done, _ = self.env.step(action)
                self.memory.store_transition(state, action, reward, state_, done)
                state = state_
                epi_reward += reward
                disc_epi_reward += disc_factor * reward
                disc_factor *= self.gamma
                if done:
                    break
            self.update()
            self.epsilon_decay(i_episode)

            if i_episode % self.log_interval == 0:
                buf = self.env.render(log_dir=None)
                self.writer.add_scalar('count/stepA', buf[0], i_episode)
            epi_rewards.append(epi_reward)
            self.writer.add_scalar('reward/raw_reward', epi_reward, i_episode)
            disc_epi_rewards.append(disc_epi_reward)
            self.writer.add_scalar('disc_reward/raw_reward', disc_epi_reward, i_episode)

            if i_episode % self.log_interval == 0 and i_episode != 0:
                lb = max(0, len(epi_rewards) - self.est_interval)
                a_reward, q_reward = np.mean(epi_rewards[lb:]), np.percentile(epi_rewards[lb:], self.q_alpha * 100)
                self.writer.add_scalar('reward/aver_reward', a_reward, i_episode)
                self.writer.add_scalar('reward/quantile_reward', q_reward, i_episode)

                disc_a_reward, disc_q_reward = np.mean(disc_epi_rewards[lb:]), np.percentile(disc_epi_rewards[lb:],
                                                                                             self.q_alpha * 100)
                self.writer.add_scalar('disc_reward/aver_reward', disc_a_reward, i_episode)
                self.writer.add_scalar('disc_reward/quantile_reward', disc_q_reward, i_episode)
                print(f'Epi:{i_episode:05d} || a_r:{a_reward:.03f} q_r:{q_reward:.03f} '
                      + f'disc_a_r:{disc_a_reward:.03f} disc_q_r:{disc_q_reward:.03f}')
                print(f'Epi:{i_episode:05d} || model Updated with lr:{self.scheduler.get_last_lr()[0]:.2e}\n')
                self.writer.add_scalar('update/epsilon', self.epsilon, i_episode)

            self.scheduler.step()

    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        batch_size = state.shape[0]
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(0, self.action_dim)
        else:
            actions_quantile_value = self.current_net.forward(state).view(batch_size, self.q_N, self.action_dim)
            q_value = actions_quantile_value[:,self.inter_idx-1,:] +  self.slope * \
                    (actions_quantile_value[:,self.inter_idx,:]-actions_quantile_value[:,self.inter_idx-1,:])
            action = torch.argmax(q_value, dim=1).item()
        return action
    
    def update(self):
        self.current_net.to(self.device)

        for i in range(1):
            batch_states, batch_actions, batch_rewards, batch_states_, batch_is_terminals = self.memory.get_transition(self.batch_size)
            batch_states, batch_actions, batch_rewards, batch_states_, batch_is_terminals = \
                batch_states.to(self.device), batch_actions.to(self.device), batch_rewards.to(self.device), \
                    batch_states_.to(self.device), batch_is_terminals.to(self.device)
            state_action_quantile_curr = self.evaluate_quantile_at_action(self.current_net(batch_states).view(-1, self.q_N, self.action_dim), batch_actions)

            with torch.no_grad():
                tmp_next = self.target_net(batch_states_).view(-1, self.q_N, self.action_dim)
                state_quantile_next = tmp_next[:,self.inter_idx-1,:] +  self.slope * (tmp_next[:,self.inter_idx,:]-tmp_next[:,self.inter_idx-1,:])
                action_next = torch.argmax(state_quantile_next, dim=1, keepdim=True)
                state_action_quantile_next = self.evaluate_quantile_at_action(tmp_next, action_next).transpose(1, 2)
                state_action_quantile_target = batch_rewards[...,None] + (1 - batch_is_terminals[...,None,None]) * self.gamma * state_action_quantile_next

            delta = state_action_quantile_target - state_action_quantile_curr
            loss = self.huber_loss(delta)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            for tar, cur in zip(self.target_net.parameters(), self.current_net.parameters()):
                tar.data.copy_(cur.data * (1.0 - self.tau) + tar.data * self.tau)

        self.current_net.to(torch.device('cpu'))

    def huber_loss(self, delta):
        element_wise_huber_loss = torch.where(delta.abs() < self.K, 0.5 * delta.pow(2) / (self.K+1e-6), (delta.abs() - 0.5 * self.K))
        element_wise_quantile_huber_loss = torch.abs(torch.tensor(self.q_taus).to(self.device).unsqueeze(dim=0).unsqueeze(dim=-1)\
                                                     - (delta.detach() < 0).float()) * element_wise_huber_loss
        batch_quantile_huber_loss = element_wise_quantile_huber_loss.sum(dim=1).mean(dim=1, keepdim=True)
        quantile_huber_loss = batch_quantile_huber_loss.mean()
        return quantile_huber_loss
    
    def evaluate_quantile_at_action(self, state_quantiles, actions):
        action_index = actions[..., None].expand(state_quantiles.shape[0], self.q_N, 1)
        return state_quantiles.gather(dim=2, index=action_index)
    
    def epsilon_decay(self, i_episode):
        self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * (self.max_episode - i_episode) / self.max_episode