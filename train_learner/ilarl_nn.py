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
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir="runs/imitation_learning")  # You can specify a folder to save logs

    expert_states, expert_actions = load_expert_trajectories(expert_file)

    expert_states = torch.tensor(np.array(expert_states), device=device)
    expert_actions = torch.tensor(np.array(expert_actions), device=device)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    il_agent = ImitationLearning(state_dim, action_dim, num_of_NNs, device=device, seed=seed)
    
    metrics = defaultdict(list)
    
    for k in range(max_iter_num):
        start_time = time.time()

        expert_traj_states = expert_states[-1]
        expert_traj_actions = expert_actions[-1]
        
        policy_states, policy_actions, policy_rewards = collect_trajectory(env, il_agent, device, max_steps)

        expert_traj_states = expert_traj_states[:expert_traj_actions.shape[0], :] 
        policy_states = policy_states[:policy_actions.shape[0], :] 
        
        reward_loss = il_agent.update_reward(expert_traj_states, expert_traj_actions, policy_states, policy_actions, args.eta)
        metrics['reward_loss'].append(reward_loss)
        writer.add_scalar('Loss/Reward Loss', reward_loss, k)
        
        z_losses = []
        for z_index in range(num_of_NNs):
            z_states, z_actions, z_rewards = collect_trajectory(env, il_agent, device, max_steps)
            z_loss = il_agent.update_z_at_index(z_states, z_actions, z_rewards, args.gamma, args.eta, z_index)
            z_losses.append(z_loss)
        metrics['z_losses'].append(z_losses)
        writer.add_scalars(f'Loss/Z Losses', {f'Z Net {i}': loss for i, loss in enumerate(z_losses)}, k)

        policy_loss, kl_div, entropy = il_agent.update_policy(policy_states, args.eta)
        metrics['policy_loss'].append(policy_loss)
        metrics['kl_divergence'].append(kl_div)
        metrics['entropy'].append(entropy)
        
        writer.add_scalar('Loss/Policy Loss', policy_loss, k)
        writer.add_scalar('Metrics/KL Divergence', kl_div, k)
        writer.add_scalar('Metrics/Entropy', entropy, k)
        
        expert_reward = il_agent.reward(torch.cat((expert_traj_states, torch.nn.functional.one_hot(expert_traj_actions, num_classes=action_dim).float()), dim=1)).mean().item()
        policy_reward = il_agent.reward(torch.cat((policy_states, torch.nn.functional.one_hot(policy_actions, num_classes=action_dim).float()), dim=1)).mean().item()
        metrics['expert_reward'].append(expert_reward)
        metrics['policy_reward'].append(policy_reward)
        
        writer.add_scalar('Reward/Expert Reward', expert_reward, k)
        writer.add_scalar('Reward/Policy Reward', policy_reward, k)
        
        q_values = il_agent.compute_q_values(policy_states)
        metrics['avg_q_value'].append(q_values.mean().item())
        writer.add_scalar('Metrics/Avg Q-value', q_values.mean().item(), k)
        
        action_probs = torch.softmax(il_agent.policy(policy_states), dim=-1)
        metrics['action_distribution'].append(action_probs.mean(dim=0).detach().numpy())
        
        z_values = torch.stack([z_net(torch.cat((policy_states, torch.nn.functional.one_hot(policy_actions, num_classes=action_dim).float()), dim=1)) for z_net in il_agent.z_networks])
        metrics['z_mean'].append(z_values.mean().item())
        metrics['z_std'].append(z_values.std().item())
        
        writer.add_scalar('Metrics/Z Mean', z_values.mean().item(), k)
        writer.add_scalar('Metrics/Z Std', z_values.std().item(), k)
        
        metrics['episodic_return'].append(policy_rewards.sum().item())
        writer.add_scalar('Metrics/Episodic Return', policy_rewards.sum().item(), k)
        
        metrics['policy_lr'].append(il_agent.policy_optimizer.param_groups[0]['lr'])
        metrics['reward_lr'].append(il_agent.reward_optimizer.param_groups[0]['lr'])

        policy_grad_norm = compute_gradient_norm(il_agent.policy)
        reward_grad_norm = compute_gradient_norm(il_agent.reward)
        metrics['policy_grad_norm'].append(policy_grad_norm)
        metrics['reward_grad_norm'].append(reward_grad_norm)
        
        writer.add_scalar('Gradients/Policy Gradient Norm', policy_grad_norm, k)
        writer.add_scalar('Gradients/Reward Gradient Norm', reward_grad_norm, k)

        end_time = time.time()
        loop_duration = end_time - start_time
        
        print(f"Iteration {k}: Reward Loss = {reward_loss:.4f}, Policy Loss = {policy_loss:.4f}, "
              f"Avg Q-value = {metrics['avg_q_value'][-1]:.4f}, Entropy = {entropy:.4f}, "
              f"Loop Duration = {loop_duration:.4f} seconds")

    return il_agent, metrics


if __name__ == "__main__":
    args = parse_arguments()
    env = create_environment(args)
    
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    il_agent = run_imitation_learning(env, args.expert_trajs, args.max_iter_num, args.num_of_NNs, device, args.seed)
