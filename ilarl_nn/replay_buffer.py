import random
from collections import deque
import torch

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        assert (next_state is None) == done, "next_state should be None if and only if done is True"
        self.buffer.append((state, action, reward, next_state, float(done)))

    def sample(self, batch_size):
        if len(self) < batch_size:
            batch = random.sample(self.buffer, len(self))
        else:
            batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        
        # Handle None values in next_state
        next_state_tensor = torch.stack([ns if ns is not None else torch.zeros_like(state[0]) for ns in next_state])
        
        return (torch.stack(state), 
                torch.tensor(action), 
                torch.tensor(reward), 
                next_state_tensor, 
                torch.tensor(done))

    def __len__(self):
        return len(self.buffer)