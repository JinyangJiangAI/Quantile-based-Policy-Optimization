import torch.nn as nn
import torch
import numpy as np


def init_weights(module: nn.Module, gain: float = 1):
    """
    Orthogonal Initialization
    """
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            module.bias.data.fill_(0.0)
    return module


class Actor(nn.Module):
    """
    Actor Network
    """
    def __init__(self, state_dim, action_dim, init_std=1e0):
        super(Actor, self).__init__()
        self.model = nn.Sequential()
        self.model.add_module(f'fc0', init_weights(nn.Linear(state_dim, action_dim, bias=False)))
        self.log_std = nn.Parameter(torch.full((action_dim,), np.log(init_std)))
        self.log_std.requires_grad = False

    def forward(self, x):
        return self.model(x)


class Critic(nn.Module):
    """
    Critic Network
    """
    def __init__(self, state_dim, emb_dim):
        super(Critic, self).__init__()
        self.model_size = [state_dim] + emb_dim + [1]
        self.model = nn.Sequential()

        for i in range(len(self.model_size) - 2):
            self.model.add_module(f'fc{i}', init_weights(nn.Linear(self.model_size[i], self.model_size[i + 1], bias=True)))
            self.model.add_module(f'act{i}', nn.Tanh())
        self.model.add_module(f'fc{len(self.model_size) - 1}', init_weights(nn.Linear(self.model_size[-2], self.model_size[-1], bias=True)))

    def forward(self, x):
        return self.model(x)


