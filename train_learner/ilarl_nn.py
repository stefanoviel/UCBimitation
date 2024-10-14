import argparse
import gym
import my_gym
from scipy import special
import os
import sys
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict
from models.ilarl_nn_models import TwoLayerNet, ImitationLearning
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard

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
    parser.add_argument('--eta', type=float, default=10, metavar='G')
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

def compute_gradient_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


def run_imitation_learning(env, expert_file, max_iter_num, num_of_NNs, device, seed=None, max_steps=10000):
    # Create a unique directory for this run
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join("runs", f"imitation_learning_{current_time}")
    os.makedirs(log_dir, exist_ok=True)

    # Initialize TensorBoard writer with the unique directory
    writer = SummaryWriter(log_dir=log_dir)

    expert_states, expert_actions = load_expert_trajectories(expert_file)

    expert_states = torch.tensor(np.array(expert_states), device=device)
    expert_actions = torch.tensor(np.array(expert_actions), device=device)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    il_agent = ImitationLearning(state_dim, action_dim, num_of_NNs, device=device, seed=seed)
    

    all_true_rewards = []
    
    for k in range(max_iter_num):
        start_time = time.time()

        expert_traj_states = expert_states[-1]
        expert_traj_actions = expert_actions[-1]
        
        policy_states, policy_actions, true_policy_rewards = collect_trajectory(env, il_agent, device, max_steps)
        all_true_rewards.append(true_policy_rewards.mean().item())

        expert_traj_states = expert_traj_states[:expert_traj_actions.shape[0], :] 
        policy_states = policy_states[:policy_actions.shape[0], :] 
        
        reward_loss = il_agent.update_reward(expert_traj_states, expert_traj_actions, policy_states, policy_actions, args.eta)
        writer.add_scalar('Loss/Reward Loss', reward_loss, k)
        
        z_losses = []
        for z_index in range(num_of_NNs):
            z_states, z_actions, _ = collect_trajectory(env, il_agent, device, max_steps)

            estimated_z_rewards = il_agent.reward(torch.cat((z_states, torch.nn.functional.one_hot(z_actions, num_classes=action_dim).float()), dim=1)) 
            z_loss = il_agent.update_z_at_index(z_states, z_actions, estimated_z_rewards, args.gamma, args.eta, z_index)
            z_losses.append(z_loss)
        writer.add_scalars(f'Loss/Z Losses', {f'Z Net {i}': loss for i, loss in enumerate(z_losses)}, k)

        policy_loss, kl_div = il_agent.update_policy(policy_states, args.eta)
        
        writer.add_scalar('Loss/Policy Loss', policy_loss, k)
        
        estimated_expert_reward = il_agent.reward(torch.cat((expert_traj_states, torch.nn.functional.one_hot(expert_traj_actions, num_classes=action_dim).float()), dim=1)).mean().item()
        estimated_policy_reward = il_agent.reward(torch.cat((policy_states, torch.nn.functional.one_hot(policy_actions, num_classes=action_dim).float()), dim=1)).mean().item()        
        writer.add_scalar('Reward/Estimated Mean Expert Reward', estimated_expert_reward, k)
        writer.add_scalar('Reward/Estimated Mean Policy Reward', estimated_policy_reward, k)
        
        q_values = il_agent.compute_q_values(policy_states)

        writer.add_scalar('Metrics/Avg Q-value', q_values.mean().item(), k)
        
        action_probs = torch.softmax(il_agent.policy(policy_states), dim=-1)
        writer.add_histogram('Action Distribution', action_probs, k)
        
        z_values = torch.stack([z_net(torch.cat((policy_states, torch.nn.functional.one_hot(policy_actions, num_classes=action_dim).float()), dim=1)) for z_net in il_agent.z_networks])        
        writer.add_scalar('Metrics/Z Mean', z_values.mean().item(), k)
        writer.add_scalar('Metrics/Z Std', z_values.std().item(), k)
        
        writer.add_scalar('Reward/True Mean policy Reward', true_policy_rewards.mean().item(), k)
        
        policy_grad_norm = compute_gradient_norm(il_agent.policy)
        reward_grad_norm = compute_gradient_norm(il_agent.reward)
        
        writer.add_scalar('Gradients/Policy Gradient Norm', policy_grad_norm, k)
        writer.add_scalar('Gradients/Reward Gradient Norm', reward_grad_norm, k)

        # Calculate the Euclidean distance between the expert and policy state means
        expert_state_mean = expert_traj_states.mean(dim=0)
        policy_state_mean = policy_states.mean(dim=0)
        
        state_distance = torch.norm(expert_state_mean - policy_state_mean).item()
        writer.add_scalar('Metrics/State Visitation Distance', state_distance, k)

        end_time = time.time()
        loop_duration = end_time - start_time
        
        print(f"Iteration {k}: Reward Loss = {reward_loss:.4f}, Policy Loss = {policy_loss:.4f}, "
              f"Avg Q-value = {q_values.mean().item():.4f}, Estimated Mean Policy reward = {estimated_policy_reward:.4f}, True Mean Episodic Return = {true_policy_rewards.mean().item():.4f}, "
              f"Loop Duration = {loop_duration:.4f} seconds")

    return il_agent, all_true_rewards


if __name__ == "__main__":
    args = parse_arguments()
    env = create_environment(args)
    
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    il_agent, all_true_rewards = run_imitation_learning(env, args.expert_trajs, args.max_iter_num, args.num_of_NNs, device, args.seed)

    # open a csv common to all run and save true rewards together with the run parameters and date
    with open("runs/true_rewards.csv", "a") as file:
        if file.tell() == 0:  # Check if the file is empty to write the header
            file.write("timestamp,env_name,noiseE,grid_type,expert_trajs,max_iter_num,num_of_NNs,seed,eta,gamma,true_rewards\n")
        file.write(f"{datetime.now()},{args.env_name},{args.noiseE},{args.grid_type},{args.expert_trajs},{args.max_iter_num},{args.num_of_NNs},{args.seed},{args.eta},{args.gamma},{np.mean(all_true_rewards)}\n")
    