import gym
import my_gym
import numpy as np
import torch
import gymnasium as gym

def create_environment(args):
    if args.env_name.startswith('Discrete'):
        env = gym.make(args.env_name, prop=args.noiseE, env_type=args.grid_type)
    else:
        env = gym.make(args.env_name)
    
    if hasattr(env, 'seed'):
        env.seed(args.seed)
    elif hasattr(env, 'reset'):
        env.reset(seed=args.seed)
    
    return env

def collect_trajectory(env, agent, device, max_steps=10000):
    states, actions, rewards = [], [], []
    
    # Handle both old and new Gym reset API
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        state, _ = reset_result
    else:
        state = reset_result
    
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
    
    for iterations in range(max_steps):
        action = agent.select_action(state_tensor)
        
        # Handle both old and new Gym step API
        step_result = env.step(action.item())
        if len(step_result) == 4:
            next_state, reward, done, _ = step_result
        elif len(step_result) == 5:
            next_state, reward, terminated, truncated, _ = step_result
            done = terminated or truncated
        else:
            raise ValueError("Unexpected number of returns from env.step()")
        
        states.append(state_tensor.cpu().numpy())
        actions.append(action.item())
        rewards.append(reward)
        
        if done:
            print("Done")
            break

        state_tensor = torch.tensor(next_state, dtype=torch.float32, device=device)
    
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)

    # Then convert them to PyTorch tensors
    return torch.tensor(states, device=device), torch.tensor(actions, device=device), torch.tensor(rewards, device=device)
