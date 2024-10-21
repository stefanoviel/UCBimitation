import gym
import my_gym
import numpy as np
import torch

def create_environment(args):
    env = gym.make(args.env_name, prop=args.noiseE, env_type=args.grid_type)
    env.seed(args.seed)
    return env

def collect_trajectory(env, agent, device, max_steps=10000):
    states, actions, rewards = [], [], []
    state, _ = env.reset()
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
    
    for iterations in range(max_steps):
        action = agent.select_action(state_tensor)
        next_state, reward, done, _ = env.step(action.item())
        
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
