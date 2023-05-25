import torch.nn as nn


def init_weights(module: nn.Module, gain: float = 1):
    """
    Orthogonal Initialization
    """
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            module.bias.data.fill_(0.0)
    return module


class ActorDiscrete(nn.Module):
    """
    Actor Network with Discrete Actions
    """
    def __init__(self, state_dim, action_dim, emb_dim):
        super(ActorDiscrete, self).__init__()
        self.model_size = [state_dim] + emb_dim + [action_dim]
        self.model = nn.Sequential()

        for i in range(len(self.model_size) - 2):
            self.model.add_module(f'fc{i}', init_weights(nn.Linear(self.model_size[i], self.model_size[i + 1], bias=True)))
            self.model.add_module(f'act{i}', nn.Tanh())
        self.model.add_module(f'fc{len(self.model_size) - 1}', init_weights(nn.Linear(self.model_size[-2], self.model_size[-1], bias=True)))
        self.model.add_module(f'act{len(self.model_size) - 1}', nn.Softmax())

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


