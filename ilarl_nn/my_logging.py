from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
import torch

def setup_logging(args):
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    
    # Create a string of selected arguments
    selected_args = [
        'seed', 'num_of_NNs', 'use_memory_replay', 'z_std_multiplier',
        'recompute_rewards', 'target_update_freq', 'eta', 'buffer_size', 'batch_size'
    ]
    args_str = '_'.join([f"{k}={getattr(args, k)}" for k in selected_args if hasattr(args, k)])
    
    if args.log_dir is None:
        log_dir = os.path.join("runs", f"imitation_learning_{args_str}_{current_time}")
    else:
        log_dir = os.path.join(args.log_dir, f"{args_str}_{current_time}")
    
    os.makedirs(log_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir)
    
    return writer

def log_iteration_summary(env, k, data, policy_loss, reward_loss, q_values, estimated_policy_reward, duration):
    if env.spec.id.startswith('Discrete'):
        true_mean_episodic_return = data['true_policy_rewards'].mean().item()
    else:
        true_mean_episodic_return = data['true_policy_rewards'].sum().item()

    print(f"Iteration {k}: "
          f"Reward Loss = {reward_loss:.4f}, "
          f"Policy Loss = {policy_loss:.4f}, "
          f"Avg Q-value = {q_values.mean().item():.4f}, "
          f"Estimated Mean Policy reward = {estimated_policy_reward:.4f}, "
          f"True Mean Episodic Return = {true_mean_episodic_return:.4f}, "
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

def log_replay_buffer_sizes(writer, il_agent, iteration):
    policy_buffer_size = il_agent.get_policy_replay_buffer_size()
    z_buffer_size = il_agent.get_z_replay_buffer_size()
    writer.add_scalar('Buffer/Policy Size', policy_buffer_size, iteration)
    writer.add_scalar('Buffer/Z Size', z_buffer_size, iteration)

def log_policy_metrics(writer, il_agent, states, k):
    # Log action probabilities distribution
    with torch.no_grad():
        action_probs = il_agent.get_action_probs(states)
        entropy = -(action_probs * torch.log(action_probs + 1e-10)).sum(dim=1).mean()
        max_prob = action_probs.max(dim=1)[0].mean()
        
        writer.add_scalar('Policy/Entropy', entropy, k)
        writer.add_scalar('Policy/Max_Action_Probability', max_prob, k)

def log_z_networks_metrics(writer, il_agent, states, actions, k):
    # compute metrics on z-networks for the recent states

    # Log z-values statistics
    z_values = torch.stack([z_net(torch.cat((states, actions), dim=1)) for z_net in il_agent.z_networks])
    z_mean = z_values.mean()
    z_std = z_values.std()
    z_max = z_values.max()
    z_min = z_values.min()
    
    writer.add_scalar('Z_Networks/Mean', z_mean, k)
    writer.add_scalar('Z_Networks/Std', z_std, k)
    writer.add_scalar('Z_Networks/Max', z_max, k)
    writer.add_scalar('Z_Networks/Min', z_min, k)
    
    # Log disagreement between z-networks
    # Disagreement measures how much the different z-networks differ in their predictions
    # For each pair of z-networks, calculate absolute difference in their predictions
    # Higher disagreement suggests more uncertainty in the z-value estimates
    pairwise_diffs = torch.pdist(z_values.squeeze())
    avg_disagreement = pairwise_diffs.mean()
    writer.add_scalar('Z_Networks/Average_Disagreement', avg_disagreement, k)

def log_network_gradients(writer, il_agent, k):
    # Log gradient norms
    for name, param in il_agent.policy.named_parameters():
        if param.grad is not None:
            writer.add_scalar(f'Gradients/Policy_{name}_norm', param.grad.norm(), k)
    
    for name, param in il_agent.reward.named_parameters():
        if param.grad is not None:
            writer.add_scalar(f'Gradients/Reward_{name}_norm', param.grad.norm(), k)
    
    for i, z_net in enumerate(il_agent.z_networks):
        for name, param in z_net.named_parameters():
            if param.grad is not None:
                writer.add_scalar(f'Gradients/Z{i}_{name}_norm', param.grad.norm(), k)

def log_expert_policy_comparison(writer, il_agent, expert_states, expert_actions, policy_states, policy_actions, k):
    # Compare state visitation
    expert_state_mean = expert_states.mean(dim=0)
    policy_state_mean = policy_states.mean(dim=0)
    state_visitation_diff = torch.norm(expert_state_mean - policy_state_mean)
    writer.add_scalar('Expert_Comparison/State_Distribution_Distance', state_visitation_diff, k)
    
    # Compare action distributions
    # Count occurrences of each action in expert trajectories, ensuring array length matches action dimension
    expert_action_dist = torch.bincount(expert_actions.flatten(), minlength=il_agent.action_dim).float()
    # Count occurrences of each action in policy trajectories, ensuring array length matches action dimension 
    policy_action_dist = torch.bincount(policy_actions.flatten(), minlength=il_agent.action_dim).float()
    # Normalize expert action counts to get probability distribution
    expert_action_dist = expert_action_dist / expert_action_dist.sum()
    # Normalize policy action counts to get probability distribution
    policy_action_dist = policy_action_dist / policy_action_dist.sum()
    # Calculate KL divergence between expert and policy action distributions
    # Small epsilon (1e-10) added to avoid numerical instability from log(0)
    action_dist_kl = torch.sum(expert_action_dist * torch.log(expert_action_dist / (policy_action_dist + 1e-10) + 1e-10))
    # Log the KL divergence to tensorboard for tracking
    writer.add_scalar('Expert_Comparison/Action_Distribution_KL', action_dist_kl, k)
