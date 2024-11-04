import gym
import numpy as np
import torch
from sac.new_sac import SAC, ReplayBuffer
from sac import core

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Environment setup
def make_env():
    return gym.make('MountainCarContinuous-v0')

# Initialize environment to get dimensions
env = make_env()
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

# Create replay buffer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, device=device)

# SAC hyperparameters
sac_kwargs = {
    'steps_per_epoch': 4000,
    'epochs': 10,
    'start_steps': 10000,
    'update_after': 1000,
    'update_every': 50,
    'batch_size': 100,
    'lr': 1e-3,
    'gamma': 0.99,
    'polyak': 0.995,
    'alpha': 0.2,
    'automatic_alpha_tuning': True,
    'device': device,
    'reinitialize': True,
}

# Initialize SAC agent
sac = SAC(
    env_fn=make_env,
    replay_buffer=replay_buffer,
    **sac_kwargs
)

# Explicitly set the test function to use the environment rewards
sac.test_fn = sac.test_agent_ori_env

# Train the agent
results = sac.learn_mujoco(print_out=True)

# Test the trained agent
test_returns = sac.test_agent_ori_env(deterministic=True)
print(f"Average test return: {test_returns:.2f}")