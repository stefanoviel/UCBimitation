import argparse
import gym
import my_gym
from scipy import special
import os
import sys
import pickle
import time
import matplotlib.pyplot as plt
import gym
import my_gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
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
    parser.add_argument('--eta', type=float, default=1e-1, metavar='G')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G')

    # parser.add_argument('--beta', type=float, default=100.0, metavar='G',
    #                     help='log std for the policy (default: -0.0)')
    # parser.add_argument('--n-expert-trajs', type=int, default=2, metavar='N')

    return parser.parse_args()

def create_environment(args):
    env = gym.make(args.env_name, prop=args.noiseE, env_type=args.grid_type)
    env.seed(args.seed)
    return env

def load_expert_trajectories(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data['states'], data['actions']


def collect_trajectory(env, agent, max_steps=10000):
    # TODO: does the number of steps should be geometric? 
    states, actions, rewards = [], [], []
    state = env.reset()
    for _ in range(max_steps):
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        
        if done:
            break
        state = next_state
    
    return states, actions, rewards


def run_imitation_learning(env, expert_file, max_iter_num, num_of_NNs, max_steps=10000):
    expert_states, expert_actions = load_expert_trajectories(expert_file)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    il_agent = ImitationLearning(state_dim, action_dim, num_of_NNs)
    
    for k in range(max_iter_num):
        # Sample expert trajectory
        expert_traj_idx = np.random.randint(len(expert_states))
        expert_traj_states = expert_states[expert_traj_idx]
        expert_traj_actions = expert_actions[expert_traj_idx]
        
        # Sample policy trajectory
        policy_states, policy_actions, policy_rewards = collect_trajectory(env, il_agent, max_steps)
        
        # Update cost function
        cost_loss = il_agent.update_cost(expert_traj_states, expert_traj_actions, policy_states, policy_actions, args.eta)
        
        # Update z networks
        for z_index in range(num_of_NNs):
            z_states, z_actions, z_rewards = collect_trajectory(env, il_agent, max_steps)
            il_agent.update_z_at_index(z_states, z_actions, z_rewards, args.gamma, args.eta, z_index)
        
        # Update policy
        policy_loss = il_agent.update_policy(policy_states, args.eta)
        

        print(f"Iteration {k}: Cost Loss = {cost_loss:.4f}, Policy Loss = {policy_loss:.4f}")
    
    return il_agent


if __name__ == "__main__":
    args = parse_arguments()
    env = create_environment(args)
    
    il_agent = run_imitation_learning(env, args.expert_trajs, args.max_iter_num, args.num_of_NNs)
    
    # You can add code here to evaluate the trained agent or save the model


# python -m train_learner.ilarl_nn --env-name DiscreteGaussianGridworld-v0  --expert-trajs assets/envDiscreteGaussianGridworld-v0type1noiseE0.0/expert_trajs/trajs16.pkl --max-iter-num 50  --grid-type 1 --noiseE 0.0 --seed 1       