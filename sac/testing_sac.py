import gym
import numpy as np
import torch
from sac.sac import SAC, ReplayBuffer
from sac import core
import time

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

# Test and visualize the trained agent
def visualize_agent(agent, env, num_episodes=5, delay=0.02):
    """
    Visualize the trained agent's performance.
    
    Args:
        agent: SAC agent
        env: Gym environment
        num_episodes: Number of episodes to run
        delay: Time delay between frames (in seconds)
    """
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        print(f"\nStarting episode {episode + 1}")
        
        while not done:
            env.render()
            time.sleep(delay)  # Add delay to make visualization viewable
            
            # Get action from agent
            action = agent.get_action(obs, deterministic=True)
            
            # Take step in environment
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            steps += 1
        
        print(f"Episode {episode + 1} finished after {steps} steps with reward {episode_reward:.2f}")
    
    env.close()

# Test the trained agent with visualization
print("\nStarting visualization of trained agent...")
visualize_agent(sac, env)

# Print final performance metrics
test_returns = sac.test_agent_ori_env(deterministic=True)
print(f"\nFinal average test return over {sac.num_test_episodes} episodes: {test_returns:.2f}")