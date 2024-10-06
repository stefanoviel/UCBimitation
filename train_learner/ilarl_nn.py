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
    parser.add_argument('--eta', type=float, default=1e-3, metavar='G')
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
    for iterations in range(max_steps):
        state_tensor = torch.FloatTensor(state).to(device)
        action = agent.select_action(state_tensor)
        next_state, reward, done, _ = env.step(action)
        
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        
        if done:
            print("Done")
            break

        state = next_state
    
    return states, actions, rewards

def plot_visited_states(states):    
    states = np.array(states)
    plt.scatter(states[:, 0], states[:, 1])
    plt.show()

def run_imitation_learning(env, expert_file, max_iter_num, num_of_NNs, device, seed=None, max_steps=10000):
    expert_states, expert_actions = load_expert_trajectories(expert_file)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    il_agent = ImitationLearning(state_dim, action_dim, num_of_NNs, device=device, seed=seed)
    
    for k in range(max_iter_num):
        start_time = time.time()
        
        expert_traj_states = expert_states[-1]
        expert_traj_actions = expert_actions[-1]

        policy_states, policy_actions, policy_rewards = collect_trajectory(env, il_agent, device, max_steps)
        
        cost_loss = il_agent.update_cost(expert_traj_states, expert_traj_actions, policy_states, policy_actions, args.eta)
        
        for z_index in range(num_of_NNs):
            z_states, z_actions, z_rewards = collect_trajectory(env, il_agent, device, max_steps) # based on current policy

            # state_action = il_agent.encode_actions_concatenate_states(z_states, z_actions)
            # z_rewards = il_agent.reward(state_action).squeeze().detach().numpy()
            
            # TODO: you can't use real rewards!!!
            il_agent.update_z_at_index(z_states, z_actions, z_rewards, args.gamma, args.eta, z_index)
        
        policy_loss = il_agent.update_policy(policy_states, args.eta)
        
        end_time = time.time()
        loop_duration = end_time - start_time
        
        print(f"Iteration {k}: Cost Loss = {cost_loss:.4f}, Policy Loss = {policy_loss:.4f}, average cost = {np.mean(policy_rewards)}, Loop Duration = {loop_duration:.4f} seconds")

        if k % 10 == 0:
            plot_visited_states(policy_states)
    
    return il_agent

if __name__ == "__main__":
    args = parse_arguments()
    env = create_environment(args)
    
    # Check if CUDA is available and choose a specific GPU
    if torch.cuda.is_available():
        # Get the first available GPU (you can also change to "cuda:1", etc. based on your requirements)
        device = torch.device("cuda:3")  # explicitly select GPU 0
    else:
        device = torch.device("cpu")  # fallback to CPU

    print(f"Using device: {device}")

    # Your imitation learning agent call
    il_agent = run_imitation_learning(env, args.expert_trajs, args.max_iter_num, args.num_of_NNs, device, args.seed)