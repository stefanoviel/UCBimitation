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
import pandas as pd
import csv
import fcntl  # For Unix-based systems

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
    parser.add_argument('--use-memory-replay', action='store_true',
                        help='use memory replay for policy updates')
    parser.add_argument('--buffer-size', type=int, default=2e5, metavar='N',
                        help='size of the replay buffer')
    parser.add_argument('--batch-size', type=int, default=2e4, metavar='N',
                        help='batch size for policy updates')
    parser.add_argument('--log-dir', type=str, default=None,
                        help='directory for tensorboard logs')
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

def setup_logging(log_dir=None, use_memory_replay=False):
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    memory_replay_str = "with_memory_replay" if use_memory_replay else "without_memory_replay"
    if log_dir is None:
        log_dir = os.path.join("runs", f"imitation_learning_{memory_replay_str}_{current_time}")
    else:
        log_dir = os.path.join(log_dir, f"{memory_replay_str}_{current_time}")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def load_and_preprocess_expert_data(expert_file, device):
    expert_states, expert_actions = load_expert_trajectories(expert_file)
    return torch.tensor(np.array(expert_states), device=device), torch.tensor(np.array(expert_actions), device=device)

def collect_iteration_data(env, il_agent, expert_states, expert_actions, device, max_steps):
    expert_traj_states = expert_states[-1]
    expert_traj_actions = expert_actions[-1]
    
    policy_states, policy_actions, true_policy_rewards = collect_trajectory(env, il_agent, device, max_steps)

    return {
        'expert_traj_states': expert_traj_states[:expert_traj_actions.shape[0], :],
        'expert_traj_actions': expert_traj_actions,
        'policy_states': policy_states[:policy_actions.shape[0], :],
        'policy_actions': policy_actions,
        'true_policy_rewards': true_policy_rewards
    }

def update_reward_and_z_networks(il_agent, data, args, writer, k, num_of_NNs, action_dim):
    reward_loss = il_agent.update_reward(data['expert_traj_states'], data['expert_traj_actions'], 
                                         data['policy_states'], data['policy_actions'], args.eta)
    writer.add_scalar('Loss/Reward Loss', reward_loss, k)
    
    z_losses = []
    for z_index in range(num_of_NNs):
        z_states, z_actions, _ = collect_trajectory(env, il_agent, device)
        estimated_z_rewards = il_agent.reward(torch.cat((z_states, torch.nn.functional.one_hot(z_actions, num_classes=action_dim).float()), dim=1)) 
        z_loss = il_agent.update_z_at_index(z_states, z_actions, estimated_z_rewards, args.gamma, args.eta, z_index)
        z_losses.append(z_loss)
    writer.add_scalars(f'Loss/Z Losses', {f'Z Net {i}': loss for i, loss in enumerate(z_losses)}, k)

    return reward_loss

def log_rewards_and_q_values(il_agent, data, writer, k, action_dim):
    estimated_expert_reward = il_agent.reward(torch.cat((data['expert_traj_states'], 
                              torch.nn.functional.one_hot(data['expert_traj_actions'], num_classes=action_dim).float()), dim=1)).mean().item()
    estimated_policy_reward = il_agent.reward(torch.cat((data['policy_states'], 
                              torch.nn.functional.one_hot(data['policy_actions'], num_classes=action_dim).float()), dim=1)).mean().item()        
    writer.add_scalar('Reward/Estimated Mean Expert Reward', estimated_expert_reward, k)
    writer.add_scalar('Reward/Estimated Mean Policy Reward', estimated_policy_reward, k)
    
    q_values = il_agent.compute_q_values(data['policy_states'])
    writer.add_scalar('Metrics/Avg Q-value', q_values.mean().item(), k)

    return q_values, estimated_policy_reward

def log_action_distribution(il_agent, policy_states, writer, k):
    action_probs = torch.softmax(il_agent.policy(policy_states), dim=-1)
    writer.add_histogram('Action Distribution', action_probs, k)

def log_z_values(il_agent, data, writer, k, action_dim):
    z_values = torch.stack([z_net(torch.cat((data['policy_states'], 
                torch.nn.functional.one_hot(data['policy_actions'], num_classes=action_dim).float()), dim=1)) 
                for z_net in il_agent.z_networks])        
    writer.add_scalar('Metrics/Z Mean', z_values.mean().item(), k)
    writer.add_scalar('Metrics/Z Std', z_values.std().item(), k)

def log_gradient_norms(il_agent, writer, k):
    policy_grad_norm = compute_gradient_norm(il_agent.policy)
    reward_grad_norm = compute_gradient_norm(il_agent.reward)
    writer.add_scalar('Gradients/Policy Gradient Norm', policy_grad_norm, k)
    writer.add_scalar('Gradients/Reward Gradient Norm', reward_grad_norm, k)

def log_state_visitation_distance(data, writer, k):
    expert_state_mean = data['expert_traj_states'].mean(dim=0)
    policy_state_mean = data['policy_states'].mean(dim=0)
    state_distance = torch.norm(expert_state_mean - policy_state_mean).item()
    writer.add_scalar('Metrics/State Visitation Distance', state_distance, k)

def log_iteration_summary(k, data, policy_loss, reward_loss, q_values, estimated_policy_reward, duration):
    print(f"Iteration {k}: "
          f"Reward Loss = {reward_loss:.4f}, "
          f"Policy Loss = {policy_loss:.4f}, "
          f"Avg Q-value = {q_values.mean().item():.4f}, "
          f"Estimated Mean Policy reward = {estimated_policy_reward:.4f}, "
          f"True Mean Episodic Return = {data['true_policy_rewards'].mean().item():.4f}, "
          f"Loop Duration = {duration:.4f} seconds")

def log_average_true_reward(writer, true_rewards, iteration):
    """
    Log the average true reward to TensorBoard.
    
    Args:
    writer (SummaryWriter): TensorBoard writer object
    true_rewards (list): List of true rewards
    iteration (int): Current iteration number
    """
    avg_true_reward = sum(true_rewards) / len(true_rewards)
    writer.add_scalar('Reward/Average True Reward', avg_true_reward, iteration)

def safe_write_csv(file_path, data, fieldnames):
    file_exists = os.path.exists(file_path)
    
    with open(file_path, 'a' if file_exists else 'w', newline='') as f:
        fcntl.flock(f, fcntl.LOCK_EX)  # Acquire an exclusive lock
        try:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(data)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)  # Release the lock

def update_policy(il_agent, iteration_data, args):
    if args.use_memory_replay:
        # Add experiences to policy replay buffer
        for i in range(len(iteration_data['policy_states'])):
            il_agent.add_policy_experience(
                iteration_data['policy_states'][i],
                iteration_data['policy_actions'][i],
                iteration_data['true_policy_rewards'][i],
                iteration_data['policy_states'][i+1] if i+1 < len(iteration_data['policy_states']) else None,
                i+1 == len(iteration_data['policy_states'])
            )

        # Update policy using replay buffer
        policy_loss, kl_div = il_agent.update_policy(args.eta)
    else:
        # Update policy using current iteration data
        policy_loss, kl_div = il_agent.update_policy_without_replay(
            iteration_data['policy_states'], 
            iteration_data['policy_actions'], 
            args.eta
        )

    return policy_loss, kl_div
        

def update_z_networks(il_agent, data, args, writer, k, num_of_NNs, action_dim):
    for z_index in range(num_of_NNs):
        if args.use_memory_replay:
            # Collect new experiences for z network
            z_states, z_actions, _ = collect_trajectory(env, il_agent, device)
            estimated_z_rewards = il_agent.reward(torch.cat((z_states, torch.nn.functional.one_hot(z_actions, num_classes=action_dim).float()), dim=1))
            for i in range(len(z_states)):
                il_agent.add_z_experience(
                    z_states[i],
                    z_actions[i],
                    estimated_z_rewards[i].item(),
                    z_states[i+1] if i+1 < len(z_states) else None,
                    i+1 == len(z_states),
                    z_index
                )
            z_loss = il_agent.update_z_at_index(None, None, None, args.gamma, args.eta, z_index)
        else:
            z_states, z_actions, _ = collect_trajectory(env, il_agent, device)
            estimated_z_rewards = il_agent.reward(torch.cat((z_states, torch.nn.functional.one_hot(z_actions, num_classes=action_dim).float()), dim=1))
            z_loss = il_agent.update_z_at_index(z_states, z_actions, estimated_z_rewards, args.gamma, args.eta, z_index)

    return z_loss


def run_imitation_learning(env, expert_file, max_iter_num, num_of_NNs, device, seed=None, max_steps=10000, 
                           use_memory_replay=False, buffer_size=None, batch_size=None, log_dir=None):
    log_dir = setup_logging(log_dir, use_memory_replay)
    writer = SummaryWriter(log_dir=log_dir)

    expert_states, expert_actions = load_and_preprocess_expert_data(expert_file, device)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    il_agent = ImitationLearning(state_dim, action_dim, num_of_NNs, device=device, seed=seed, 
                                 use_memory_replay=use_memory_replay, buffer_size=buffer_size, batch_size=batch_size)

    all_true_rewards = []
    
    for k in range(max_iter_num):
        start_time = time.time()

        iteration_data = collect_iteration_data(env, il_agent, expert_states, expert_actions, device, max_steps)
        all_true_rewards.append(iteration_data['true_policy_rewards'].mean().item())
        log_average_true_reward(writer, all_true_rewards, k)

        # update policy here
        policy_loss, kl_div = update_policy(il_agent, iteration_data, args)
        reward_loss = il_agent.update_reward(iteration_data['expert_traj_states'], iteration_data['expert_traj_actions'], 
                                             iteration_data['policy_states'], iteration_data['policy_actions'], args.eta)
        z_loss = update_z_networks(il_agent, iteration_data, args, writer, k, num_of_NNs, action_dim)

        q_values, estimated_policy_reward = log_rewards_and_q_values(il_agent, iteration_data, writer, k, action_dim)
        log_action_distribution(il_agent, iteration_data['policy_states'], writer, k)
        log_z_values(il_agent, iteration_data, writer, k, action_dim)
        log_gradient_norms(il_agent, writer, k)
        log_state_visitation_distance(iteration_data, writer, k)

        end_time = time.time()
        log_iteration_summary(k, iteration_data, policy_loss, reward_loss, q_values, estimated_policy_reward, end_time - start_time)

    return il_agent, all_true_rewards

def prepare_csv_data(args, all_true_rewards):
    run_id = datetime.now().strftime('%Y%m%d-%H%M%S')
    data_to_write = []
    for i in range(args.max_iter_num):
        row = {
            'iteration': i,
            'true_reward': all_true_rewards[i],
            'run_id': run_id
        }
        row.update({arg: str(value) for arg, value in vars(args).items()})
        data_to_write.append(row)
    return data_to_write, list(data_to_write[0].keys())

def save_results(args, all_true_rewards):
    data_to_write, fieldnames = prepare_csv_data(args, all_true_rewards)
    log_file_path = os.path.join(args.log_dir, "true_rewards.csv") if args.log_dir else "runs/true_rewards.csv"
    safe_write_csv(log_file_path, data_to_write, fieldnames)
    print(f"True rewards and run parameters saved to {log_file_path}")


if __name__ == "__main__":
    args = parse_arguments()
    env = create_environment(args)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    il_agent, all_true_rewards = run_imitation_learning(
        env, args.expert_trajs, args.max_iter_num, args.num_of_NNs, device, args.seed,
        use_memory_replay=args.use_memory_replay,
        buffer_size=int(args.buffer_size),
        batch_size=int(args.batch_size),
        log_dir=args.log_dir
    )

    save_results(args, all_true_rewards)