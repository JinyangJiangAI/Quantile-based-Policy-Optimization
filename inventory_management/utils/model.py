import torch
import torch.nn as nn


class ActorDiscrete(nn.Module):
    """
    Actor Network with Discrete Actions
    """
    def __init__(self, state_dim, action_dim):
        super(ActorDiscrete, self).__init__()
        self.conv_block = nn.Sequential(nn.Conv1d(in_channels=state_dim[1], out_channels=32, kernel_size=3, padding=0), nn.Tanh(),
                                        nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=0), nn.Tanh())
        # self.conv_block = nn.Sequential(nn.Conv1d(in_channels=state_dim[1], out_channels=64, kernel_size=3, padding=0), nn.Tanh())
        action_dim = action_dim.tolist()
        self.fc_block = nn.Sequential()
        for i, d in enumerate(action_dim):
            self.fc_block.add_module('fc_'+str(i),nn.Linear(in_features=64 * (state_dim[0] - 4), out_features=d))

    def forward(self, x):
        is_batched = x.dim() == 3
        if not is_batched:
            x = x.unsqueeze(0)
        x = self.conv_block(x)
        x = torch.reshape(x, (x.shape[0], -1))
        out = []
        for fc in self.fc_block:
            out.append(torch.softmax(fc(x), dim=-1))
        if not is_batched:
            out = [x.squeeze(0) for x in out]
        return torch.stack(out,dim=-2)


class Critic(nn.Module):
    """
    Critic Network
    """
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.conv_block = nn.Sequential(nn.Conv1d(in_channels=state_dim[1], out_channels=32, kernel_size=3, padding=0), nn.Tanh(),
                                        nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=0), nn.Tanh())
        # self.conv_block = nn.Sequential(nn.Conv1d(in_channels=state_dim[1], out_channels=64, kernel_size=3, padding=0), nn.Tanh())
        self.fc = nn.Linear(in_features=64 * (state_dim[0] - 4), out_features=1)

    def forward(self, x):
        is_batched = x.dim() == 3
        if not is_batched:
            x = x.unsqueeze(0)
        x = self.conv_block(x)
        x = torch.reshape(x, (x.shape[0], -1))
        x = self.fc(x)
        if not is_batched:
            x = x.squeeze(0)
        return x

