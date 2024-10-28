from ilarl_nn.logging import setup_logging, log_rewards_and_q_values, log_iteration_summary, log_average_true_reward, log_replay_buffer_sizes
from ilarl_nn.data_handling import load_and_preprocess_expert_data
from ilarl_nn.environment import collect_trajectory
from ilarl_nn.utils import prepare_csv_data, safe_write_csv
from ilarl_nn.ilarl_nn_models import TwoLayerNet, ImitationLearning
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard
import torch
import time




def update_z_networks(il_agent, args, num_of_NNs, action_dim, env, device):
    z_loss = 0
    for z_index in range(num_of_NNs):
        # Collect new experiences and update z network directly
        z_states, z_actions, _ = collect_trajectory(env, il_agent, device)
        estimated_z_rewards = il_agent.reward(torch.cat((z_states, torch.nn.functional.one_hot(z_actions, num_classes=action_dim).float()), dim=1))
        z_loss += il_agent.update_z_without_replay(z_states, z_actions, estimated_z_rewards, args.gamma, args.eta, z_index)
    return z_loss / num_of_NNs

def run_imitation_learning(env, expert_file, max_iter_num, num_of_NNs, device, args, z_std_multiplier, seed=None, max_steps=10000):
    writer = setup_logging(args)

    expert_states, expert_actions = load_and_preprocess_expert_data(expert_file, device)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    il_agent = ImitationLearning(
        state_dim, 
        action_dim, 
        num_of_NNs, 
        device=device, 
        seed=seed,
        z_std_multiplier=z_std_multiplier
    )

    all_true_rewards = []

    for k in range(max_iter_num):
        start_time = time.time()

        policy_states, policy_actions, true_policy_rewards = collect_trajectory(env, il_agent, device, max_steps)

        iteration_data = {
            'expert_traj_states': expert_states,
            'expert_traj_actions': expert_actions,
            'policy_states': policy_states,
            'policy_actions': policy_actions,
            'true_policy_rewards': true_policy_rewards
        }
        
        if args.env_name.startswith('Discrete'):
            all_true_rewards.append(iteration_data['true_policy_rewards'].mean().item())
        else:
            all_true_rewards.append(iteration_data['true_policy_rewards'].sum().item())
        log_average_true_reward(writer, all_true_rewards, k)


        policy_loss = il_agent.update_policy(
            iteration_data['policy_states'], 
            iteration_data['policy_actions'], 
            args.eta
        )

        reward_loss = il_agent.update_reward(
            iteration_data['expert_traj_states'], 
            iteration_data['expert_traj_actions'], 
            iteration_data['policy_states'], 
            iteration_data['policy_actions'], 
            args.eta
        )

        z_loss = update_z_networks(il_agent, args, num_of_NNs, action_dim, env, device)
        z_variance = il_agent.compute_z_variance()
        writer.add_scalar('Metrics/Z Variance', z_variance, k)

        q_values, estimated_policy_reward = log_rewards_and_q_values(il_agent, iteration_data, writer, k, action_dim)

        end_time = time.time()
        log_iteration_summary(env, k, iteration_data, policy_loss, reward_loss, q_values, estimated_policy_reward, end_time - start_time)

    return il_agent, all_true_rewards
