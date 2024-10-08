import argparse
import gym
import my_gym
from scipy import special
import os
import sys
import pickle
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.ilarl_nn_models import TwoLayerNet, ImitationLearning


def parse_arguments():
    parser = argparse.ArgumentParser(description='UCB')
    parser.add_argument('--env-name', default="DiscreteGaussianGridworld-v0", metavar='G',
                    help='name of the environment to run')
    parser.add_argument('--noiseE', type=float, default=0.0, metavar='G', help='probability of choosing a random action')
    parser.add_argument('--grid-type', type=int, default=None, metavar='N', help='1 easier, 0 harder, check environment for more details')
    parser.add_argument('--expert-trajs', metavar='G', help='path to expert data')
    parser.add_argument('--max-iter-num', type=int, default=500, metavar='N',
                        help='maximal number of run of the algorithm')
    parser.add_argument('--num-of-NNs', type=int, default=5, metavar='N',
                        help='number of neural networks to use')
    parser.add_argument('--seed', type=int, default=1, metavar='N')
    parser.add_argument('--eta', type=float, default=1, metavar='G')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G')
    return parser.parse_args()

def create_environment(args):
    env = gym.make(args.env_name, prop=args.noiseE, env_type=args.grid_type)
    env.seed(args.seed)
    return env

def load_expert_trajectories(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data['states'], data['actions']

def collect_trajectory(env, agent, device, max_steps=10000):
    states, actions, rewards = [], [], []
    state, _ = env.reset()
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
    for iterations in range(max_steps):
        action = agent.select_action(state_tensor)
        next_state, reward, done, _ = env.step(action.item())
        
        states.append(state_tensor.cpu().numpy())  # TODO: they go on cpu and then come back to gpu
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

def plot_visited_states(states):    
    states_cpu = states.cpu().numpy()
    plt.scatter(states_cpu[:, 0], states_cpu[:, 1])
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.show()

def run_imitation_learning(env, expert_file, max_iter_num, num_of_NNs, device, seed=None, max_steps=10000):
    expert_states, expert_actions = load_expert_trajectories(expert_file)
    expert_states = torch.tensor(expert_states, device=device)
    expert_actions = torch.tensor(expert_actions, device=device)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    il_agent = ImitationLearning(state_dim, action_dim, num_of_NNs, device=device, seed=seed)
    
    for k in range(max_iter_num):
        start_time = time.time()
        
        # TODO: consider more than one expert trajectory
        expert_traj_states = expert_states[-1]
        expert_traj_actions = expert_actions[-1]
        
        policy_states, policy_actions, policy_rewards = collect_trajectory(env, il_agent, device, max_steps)
        
        reward_loss = il_agent.update_reward(expert_traj_states, expert_traj_actions, policy_states, policy_actions, args.eta)
        
        for z_index in range(num_of_NNs):
            z_states, z_actions, z_rewards = collect_trajectory(env, il_agent, device, max_steps)
            # TODO: you can't use real rewards here, because you don't know them
            il_agent.update_z_at_index(z_states, z_actions, z_rewards, args.gamma, args.eta, z_index)
        

        policy_loss = il_agent.update_policy(policy_states, args.eta)
        
        end_time = time.time()
        loop_duration = end_time - start_time
        
        print(f"Iteration {k}: Reward Loss = {reward_loss:.4f}, Policy Loss = {policy_loss}, average reward = {policy_rewards.mean().item()}, Loop Duration = {loop_duration:.4f} seconds")

        if k % 5 == 0 and k > 0:
            plot_visited_states(policy_states)
    
    return il_agent


if __name__ == "__main__":
    args = parse_arguments()
    env = create_environment(args)
    
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    il_agent = run_imitation_learning(env, args.expert_trajs, args.max_iter_num, args.num_of_NNs, device, args.seed)