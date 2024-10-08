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
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
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

def compute_gradient_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm
    
def plot_metrics(metrics, iteration):
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(f'Training Metrics at Iteration {iteration}')
    
    axs[0, 0].plot(metrics['expert_reward'], label='Expert')
    axs[0, 0].plot(metrics['policy_reward'], label='Policy')
    axs[0, 0].set_title('Rewards')
    axs[0, 0].legend()
    
    axs[0, 1].plot(metrics['policy_loss'])
    axs[0, 1].set_title('Policy Loss')
    
    axs[0, 2].plot(metrics['reward_loss'])
    axs[0, 2].set_title('Reward Loss')
    
    axs[1, 0].plot(metrics['kl_divergence'])
    axs[1, 0].set_title('KL Divergence')
    
    axs[1, 1].plot(metrics['entropy'])
    axs[1, 1].set_title('Policy Entropy')
    
    axs[1, 2].plot(metrics['avg_q_value'])
    axs[1, 2].set_title('Average Q-value')
    
    axs[2, 0].plot(metrics['z_mean'], label='Mean')
    axs[2, 0].plot(metrics['z_std'], label='Std')
    axs[2, 0].set_title('Z-value Statistics')
    axs[2, 0].legend()
    
    axs[2, 1].plot(metrics['episodic_return'])
    axs[2, 1].set_title('Episodic Return')
    
    action_dist = np.array(metrics['action_distribution'])
    for i in range(action_dist.shape[1]):
        axs[2, 2].plot(action_dist[:, i], label=f'Action {i}')
    axs[2, 2].set_title('Action Distribution')
    axs[2, 2].legend()
    
    plt.tight_layout()
    plt.show()


def run_imitation_learning(env, expert_file, max_iter_num, num_of_NNs, device, seed=None, max_steps=10000):
    expert_states, expert_actions = load_expert_trajectories(expert_file)

    expert_states = torch.tensor(np.array(expert_states), device=device)
    expert_actions = torch.tensor(np.array(expert_actions), device=device)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    il_agent = ImitationLearning(state_dim, action_dim, num_of_NNs, device=device, seed=seed)
    
    # Initialize metrics tracking
    metrics = defaultdict(list)
    
    for k in range(max_iter_num):
        start_time = time.time()


        # TODO: is one trajectory enough? do we expect to learn from just one? 
        expert_traj_states = expert_states[-1]
        expert_traj_actions = expert_actions[-1]
        
        policy_states, policy_actions, policy_rewards = collect_trajectory(env, il_agent, device, max_steps)

        # if we terminated there will be one more state than action because for the last state we don't have an action
        # TODO: check correctness
        expert_traj_states = expert_traj_states[:expert_traj_actions.shape[0], :] 
        policy_states = policy_states[:policy_actions.shape[0], :] 
        
        # Update reward network and track loss
        reward_loss = il_agent.update_reward(expert_traj_states, expert_traj_actions, policy_states, policy_actions, args.eta)
        metrics['reward_loss'].append(reward_loss)
        
        # Update Z-networks and track losses
        z_losses = []
        for z_index in range(num_of_NNs):
            z_states, z_actions, z_rewards = collect_trajectory(env, il_agent, device, max_steps)
            z_loss = il_agent.update_z_at_index(z_states, z_actions, z_rewards, args.gamma, args.eta, z_index)
            z_losses.append(z_loss)
        metrics['z_losses'].append(z_losses)
        
        # Update policy and track metrics
        policy_loss, kl_div, entropy = il_agent.update_policy(policy_states, args.eta)
        metrics['policy_loss'].append(policy_loss)
        metrics['kl_divergence'].append(kl_div)
        metrics['entropy'].append(entropy)
        
        # Track rewards
        expert_reward = il_agent.reward(torch.cat((expert_traj_states, torch.nn.functional.one_hot(expert_traj_actions, num_classes=action_dim).float()), dim=1)).mean().item()
        policy_reward = il_agent.reward(torch.cat((policy_states, torch.nn.functional.one_hot(policy_actions, num_classes=action_dim).float()), dim=1)).mean().item()
        metrics['expert_reward'].append(expert_reward)
        metrics['policy_reward'].append(policy_reward)
        
        # Track Q-values
        q_values = il_agent.compute_q_values(policy_states)
        metrics['avg_q_value'].append(q_values.mean().item())
        
        # Track action distribution
        action_probs = torch.softmax(il_agent.policy(policy_states), dim=-1)
        metrics['action_distribution'].append(action_probs.mean(dim=0).detach().numpy())
        
        # Track Z-value statistics
        z_values = torch.stack([z_net(torch.cat((policy_states, torch.nn.functional.one_hot(policy_actions, num_classes=action_dim).float()), dim=1)) for z_net in il_agent.z_networks])
        metrics['z_mean'].append(z_values.mean().item())
        metrics['z_std'].append(z_values.std().item())
        
        # Track episodic return
        metrics['episodic_return'].append(policy_rewards.sum().item())
        
        # Track learning rates
        metrics['policy_lr'].append(il_agent.policy_optimizer.param_groups[0]['lr'])
        metrics['reward_lr'].append(il_agent.reward_optimizer.param_groups[0]['lr'])
        
        # Track gradient norms
        policy_grad_norm = compute_gradient_norm(il_agent.policy)
        reward_grad_norm = compute_gradient_norm(il_agent.reward)
        metrics['policy_grad_norm'].append(policy_grad_norm)
        metrics['reward_grad_norm'].append(reward_grad_norm)
        
        end_time = time.time()
        loop_duration = end_time - start_time
        
        print(f"Iteration {k}: Reward Loss = {reward_loss:.4f}, Policy Loss = {policy_loss:.4f}, "
              f"Avg Q-value = {metrics['avg_q_value'][-1]:.4f}, Entropy = {entropy:.4f}, "
              f"Loop Duration = {loop_duration:.4f} seconds")

        if k % 5 == 0 and k > 0:
            plot_metrics(metrics, k)
    
    return il_agent, metrics


if __name__ == "__main__":
    args = parse_arguments()
    env = create_environment(args)
    
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    il_agent = run_imitation_learning(env, args.expert_trajs, args.max_iter_num, args.num_of_NNs, device, args.seed)