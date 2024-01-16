import numpy as np
import torch
from torch.distributions import Categorical
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

from utils import ActorDiscrete


class SPSA(object):
    """
    Simultaneous Perturbation Stochastic Approximation
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
        self.actor_plus = ActorDiscrete(state_dim, action_dim, args.emb_dim)
        self.actor_minus = ActorDiscrete(state_dim, action_dim, args.emb_dim)

        self.optimizer = Adam(self.actor.parameters(), args.lr, eps=1e-5)
        self.scheduler = StepLR(self.optimizer, step_size=args.lr_decay_freq, gamma=args.lr_decay_rate)
        self.MSELoss = torch.nn.MSELoss()

        self.writer = SummaryWriter(log_dir=args.path)

        self.spsa_batch = args.spsa_batch
        self.spsa_params = {'c':args.perturb_c, 'gamma':args.perturb_gamma}
        self.delta = []
        self.c = None

    def generate_delta(self):
        for p in self.actor.parameters():
            self.delta.append(2 * (torch.bernoulli(0.5 * torch.ones_like(p)) - 0.5))

    def model_perturb(self):
        self.actor_plus.load_state_dict(self.actor.state_dict())
        self.actor_minus.load_state_dict(self.actor.state_dict())
        with torch.no_grad():
            for p1, p2, d in zip(self.actor_plus.parameters(), self.actor_minus.parameters(), self.delta):
                p1 += self.c * d
                p2 -= self.c * d

    def train(self):
        for i_episode in range(0, self.max_episode+1, self.spsa_batch*2):
            self.c = self.spsa_params['c'] / ((i_episode//(self.spsa_batch*2))+1) ** self.spsa_params['gamma']
            self.generate_delta()
            self.model_perturb()

            if i_episode % self.log_interval == 0 and i_episode != 0:
                disc_epi_rewards, accuracies = self.evaluate(self.est_interval, 0)
                disc_a_reward, disc_q_reward = np.mean(disc_epi_rewards), np.percentile(disc_epi_rewards, self.q_alpha*100)
                self.writer.add_scalar('disc_reward/aver_reward', disc_a_reward, i_episode)
                self.writer.add_scalar('disc_reward/quantile_reward', disc_q_reward, i_episode)

                for i, (dr, a) in enumerate(zip(disc_epi_rewards, accuracies)):
                    self.writer.add_scalar('disc_reward/raw_reward', dr, i_episode+i)
                    self.writer.add_scalar('acc/total_accuracy', a, i_episode+i)

                print(f'Epi:{i_episode:05d} || disc_a_r:{disc_a_reward:.03f} disc_q_r:{disc_q_reward:.03f}')
                print(f'Epi:{i_episode:05d} || model Updated with lr:{self.scheduler.get_last_lr()[0]:.2e} perturb_size:{self.c:.2e}\n')

            disc_epi_rewards_plus, _ = self.evaluate(self.spsa_batch, 1)
            disc_epi_rewards_minus, _ = self.evaluate(self.spsa_batch, 2)

            f_plus = np.percentile(disc_epi_rewards_plus, 100*self.q_alpha)
            f_minus = np.percentile(disc_epi_rewards_minus, 100*self.q_alpha)
            diff = f_plus - f_minus
            self.update(diff.item())

            for i in range(2*self.spsa_batch):
                self.scheduler.step()

    def choose_action(self, state, mode=0):
        if mode == 0:
            actor = self.actor
        elif mode == 1:
            actor = self.actor_plus
        else:
            actor = self.actor_minus
        state = torch.from_numpy(state).float()
        action_probs = actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.detach().data.numpy()

    def evaluate(self, max_episode, mode=0):
        disc_epi_rewards, accuracies = [], []
        for i_episode in range(1, max_episode + 1):
            disc_epi_reward, disc_factor, state = 0, 1, self.env.reset()
            while True:
                action = self.choose_action(state.flatten(), mode)
                state, reward, done, _ = self.env.step(action)
                disc_epi_reward += disc_factor * reward
                disc_factor *= self.gamma
                if done:
                    break
            disc_epi_rewards.append(disc_epi_reward)
            freq = self.env.render()
            accuracies.append(freq[0])
        return disc_epi_rewards, accuracies

    def update(self, diff_f):
        self.actor.to(self.device)

        self.optimizer.zero_grad()
        for p, d in zip(self.actor.parameters(), self.delta):
            p.grad = - diff_f / (2 * self.c * d.to(self.device))
        self.optimizer.step()

        self.delta.clear()
        self.actor.to(torch.device('cpu'))