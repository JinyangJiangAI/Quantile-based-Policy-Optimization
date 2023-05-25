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