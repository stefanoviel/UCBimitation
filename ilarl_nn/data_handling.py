import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

def load_expert_trajectories(file_path):
    with open(file_path, 'rb') as f:
        expert_trajectories = pickle.load(f)
    return expert_trajectories


def load_and_preprocess_expert_data(expert_file, device):
    with open(expert_file, 'rb') as file:
        data = pickle.load(file)
    expert_states = data['states']
    expert_actions = data['actions']

    return torch.tensor(np.array(expert_states), device=device), torch.tensor(np.array(expert_actions), device=device)


def create_data_loader(states, actions, rewards, batch_size):
    dataset = TensorDataset(states, actions, rewards)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Use this in your training loop
