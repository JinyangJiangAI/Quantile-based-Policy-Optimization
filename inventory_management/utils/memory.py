import numpy as np
import torch


class Memory(object):
    """
    Memory
    """
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.values = []
        self.last_values = []
        self.t = []

    def clear(self):
        self.actions.clear()
        self.states.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.is_terminals.clear()
        self.values.clear()
        self.last_values.clear()
        self.t.clear()

    def get_len(self):
        return len(self.is_terminals)


class CircularMemory(object):
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.memory = [None] * memory_size
        self.idx = 0

    def add(self, np_array):
        self.memory[self.idx % self.memory_size] = np_array
        self.idx += 1

    def get(self, idx):
        if self.memory[idx] is not None:
            return self.memory[idx]
        else:
            return None


class QMemory(object):
    """
    QMemory
    """

    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.actions = CircularMemory(memory_size)
        self.states = CircularMemory(memory_size)
        self.states_ = CircularMemory(memory_size)
        self.rewards = CircularMemory(memory_size)
        self.is_terminals = CircularMemory(memory_size)

    def store_transition(self, state, action, reward, state_, is_terminal):
        for s, a, r, s_, d in zip(state, action, reward, state_, is_terminal):
            self.actions.add(a)
            self.states.add(s)
            self.states_.add(s_)
            self.rewards.add(r)
            d = 1 if d else 0
            self.is_terminals.add(d)

    def get_transition(self, batch_size):
        idx = np.random.randint(0, min(self.memory_size, self.actions.idx), size=batch_size)
        batch_actions = torch.from_numpy(
            np.array([self.actions.get(i) for i in idx if self.actions.get(i) is not None])).long()
        batch_states = torch.from_numpy(
            np.array([self.states.get(i) for i in idx if self.states.get(i) is not None])).float()
        batch_states_ = torch.from_numpy(
            np.array([self.states_.get(i) for i in idx if self.states_.get(i) is not None])).float()
        batch_rewards = torch.from_numpy(
            np.array([[self.rewards.get(i)] for i in idx if self.rewards.get(i) is not None])).float()
        batch_is_terminals = torch.from_numpy(
            np.array([self.is_terminals.get(i) for i in idx if self.is_terminals.get(i) is not None])).float()
        return batch_states, batch_actions, batch_rewards, batch_states_, batch_is_terminals