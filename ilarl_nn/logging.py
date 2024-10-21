from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
import torch

def setup_logging(log_dir=None, use_memory_replay=False, seed=None, num_of_NNs=None):
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    memory_replay_str = "with_memory_replay" if use_memory_replay else "without_memory_replay"
    seed_str = f"seed{seed}" if seed is not None else ""
    nn_str = f"nn{num_of_NNs}" if num_of_NNs is not None else ""
    
    if log_dir is None:
        log_dir = os.path.join("runs", f"imitation_learning_{memory_replay_str}_{seed_str}_{nn_str}_{current_time}")
    else:
        log_dir = os.path.join(log_dir, f"{memory_replay_str}_{seed_str}_{nn_str}_{current_time}")
    
    os.makedirs(log_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir)
    
    return writer

def log_iteration_summary(k, data, policy_loss, reward_loss, q_values, estimated_policy_reward, duration):
    print(f"Iteration {k}: "
          f"Reward Loss = {reward_loss:.4f}, "
          f"Policy Loss = {policy_loss:.4f}, "
          f"Avg Q-value = {q_values.mean().item():.4f}, "
          f"Estimated Mean Policy reward = {estimated_policy_reward:.4f}, "
          f"True Mean Episodic Return = {data['true_policy_rewards'].mean().item():.4f}, "
          f"Loop Duration = {duration:.4f} seconds")

def log_average_true_reward(writer, true_rewards, iteration):
    avg_reward = sum(true_rewards[-100:]) / min(len(true_rewards), 100)
    writer.add_scalar("Reward/Mean True Reward", avg_reward, iteration)

def log_rewards_and_q_values(il_agent, data, writer, k, action_dim):
    # Handle potential mismatch for expert data
    expert_states = data['expert_traj_states']
    expert_actions = data['expert_traj_actions']
    if expert_states.shape[0] != expert_actions.shape[0]:
        expert_states = expert_states[:-1]

    # Handle potential mismatch for policy data
    policy_states = data['policy_states']
    policy_actions = data['policy_actions']
    if policy_states.shape[0] != policy_actions.shape[0]:
        policy_states = policy_states[:-1]

    # Calculate estimated expert reward
    expert_actions_one_hot = torch.nn.functional.one_hot(expert_actions, num_classes=action_dim).float()
    estimated_expert_reward = il_agent.reward(torch.cat((expert_states, expert_actions_one_hot), dim=1)).mean().item()

    # Calculate estimated policy reward
    policy_actions_one_hot = torch.nn.functional.one_hot(policy_actions, num_classes=action_dim).float()
    estimated_policy_reward = il_agent.reward(torch.cat((policy_states, policy_actions_one_hot), dim=1)).mean().item()

    writer.add_scalar('Reward/Estimated Mean Expert Reward', estimated_expert_reward, k)
    writer.add_scalar('Reward/Estimated Mean Policy Reward', estimated_policy_reward, k)
    
    # Compute Q-values using the potentially trimmed policy_states
    q_values = il_agent.compute_q_values(policy_states)
    writer.add_scalar('Metrics/Avg Q-value', q_values.mean().item(), k)

    return q_values, estimated_policy_reward
