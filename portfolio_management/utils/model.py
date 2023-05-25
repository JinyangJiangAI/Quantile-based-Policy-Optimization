import torch
import torch.nn as nn
import numpy as np


def init_weights(module: nn.Module, gain: float = 1):
    """
    Orthogonal Initialization
    """
    if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            module.bias.data.fill_(0.0)
    return module


class ActorContinuous(nn.Module):
    """
    Actor Network with Continuous Actions
    """
    def __init__(self, state_dim, action_dim, init_var=1.):
        super(ActorContinuous, self).__init__()
        self.fc_block = nn.Sequential(init_weights(nn.Linear(in_features=state_dim, out_features=64)), nn.Tanh(),
                                      init_weights(nn.Linear(in_features=64, out_features=64)), nn.Tanh(),
                                      init_weights(nn.Linear(in_features=64, out_features=action_dim)))
        cov = torch.full((action_dim, action_dim), 0.)
        for i in range(action_dim):
            cov[i,i] = init_var
        self.cov = nn.Parameter(cov)

    def forward(self, x):
        x = self.fc_block(x)
        return x
        

class Critic(nn.Module):
    """
    Critic Network
    """
    def __init__(self,state_dim):
        super(Critic, self).__init__()
        self.fc_block = nn.Sequential(init_weights(nn.Linear(in_features=state_dim, out_features=64)), nn.Tanh(),
                                      init_weights(nn.Linear(in_features=64, out_features=64)), nn.Tanh(),
                                      init_weights(nn.Linear(in_features=64, out_features=1)))

    def forward(self, x):
        x = self.fc_block(x)
        return x

