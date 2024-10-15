import random
from collections import deque
import torch

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, float(done)))

    def sample(self, batch_size):
        if len(self) < batch_size:
            batch = random.sample(self.buffer, len(self))
        else:
            batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (torch.stack(state), 
                torch.tensor(action), 
                torch.tensor(reward), 
                torch.stack(next_state), 
                torch.tensor(done))

    def __len__(self):
        return len(self.buffer)
