import torch
import torch.optim as optim
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


class TwoLayerNet(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(TwoLayerNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 64)
        self.fc2 = torch.nn.Linear(64, output_size)
    
    def forward(self, x):
        x = x.float() 
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ImitationLearning:
    def __init__(self, state_dim, action_dim, num_of_NNs, learning_rate=1e-3, device='cpu', seed=None, 
                 z_std_multiplier=1.0):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_of_NNs = num_of_NNs
        self.device = device
        self.z_std_multiplier = z_std_multiplier

        if seed is not None:
            torch.manual_seed(seed)
            if str(device).startswith('cuda'):
                torch.cuda.manual_seed(seed)
        
        self.policy = TwoLayerNet(state_dim, action_dim).to(device)
        self.reward = TwoLayerNet(state_dim + action_dim, 1).to(device)
        self.z_networks = [TwoLayerNet(state_dim + action_dim, 1).to(device) for _ in range(num_of_NNs)]
        
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.reward_optimizer = optim.Adam(self.reward.parameters(), lr=learning_rate)
        self.z_optimizers = [optim.Adam(net.parameters(), lr=learning_rate) for net in self.z_networks]
        
        self.fixed_sa_pairs = None
        self.initialize_fixed_sa_pairs(100)

        # Add these new parameters
        self.reward_scale = 1.0  # Adjustable reward scaling
        
    def initialize_fixed_sa_pairs(self, num_pairs):
        """
        Initialize a set of fixed state-action pairs for computing the variance of the z-values.
        """
        states = torch.rand((num_pairs, self.state_dim), device=self.device)
        actions = torch.randint(0, self.action_dim, (num_pairs,), device=self.device)
        actions_one_hot = torch.nn.functional.one_hot(actions, num_classes=self.action_dim)
        self.fixed_sa_pairs = torch.cat((states, actions_one_hot.float()), dim=1)
    
    def select_action(self, state):
        with torch.no_grad():
            logits = self.policy(state)

            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print("state", state)
                print("logits", logits)
                print("Warning: NaN or Inf detected in logits!")
                
            action_probs = torch.softmax(logits, dim=-1)
            action = torch.multinomial(action_probs, 1)
        return action

    def update_reward(self, expert_states, expert_actions, policy_states, policy_actions, eta):

        # Remove the last state and action if they don't match
        if expert_states.shape[0] != expert_actions.shape[0]:
            expert_states = expert_states[:-1]
        if policy_states.shape[0] != policy_actions.shape[0]:
            policy_states = policy_states[:-1]

        expert_actions_one_hot = torch.nn.functional.one_hot(expert_actions, num_classes=self.action_dim)
        policy_actions_one_hot = torch.nn.functional.one_hot(policy_actions, num_classes=self.action_dim)
    
        expert_sa = torch.cat((expert_states, expert_actions_one_hot), dim=1)
        policy_sa = torch.cat((policy_states, policy_actions_one_hot), dim=1)
        
        expert_reward = self.reward(expert_sa).mean()
        policy_reward = self.reward(policy_sa).mean()

        loss = policy_reward - expert_reward
        self.reward_optimizer.zero_grad()
        loss.backward()
        self.reward_optimizer.step()
        
        return loss.item()
    
    def update_z_networks(self, states, actions, estimated_rewards, gamma, eta, z_index):
        actions_one_hot = torch.nn.functional.one_hot(actions, num_classes=self.action_dim)
        sa = torch.cat((states, actions_one_hot.float()), dim=1)

        z_net = self.z_networks[z_index]
        z_opt = self.z_optimizers[z_index]
        
        z_values = z_net(sa).squeeze()
        
        discounted_future_rewards = torch.zeros_like(estimated_rewards)
        running_sum = 0
        for t in reversed(range(len(estimated_rewards))):
            running_sum = estimated_rewards[t] + gamma * running_sum
            discounted_future_rewards[t] = running_sum
        
        # Normalize the discounted future rewards
        discounted_future_rewards = (discounted_future_rewards - discounted_future_rewards.mean()) / (discounted_future_rewards.std() + 1e-8)   
        
        loss = torch.mean((z_values - discounted_future_rewards)**2)
        
        z_opt.zero_grad()
        loss.backward()
        z_opt.step()

        return loss.item()

    def compute_q_values(self, states):
        states_expanded = states.unsqueeze(1).repeat(1, self.action_dim, 1)
        actions = torch.eye(self.action_dim, device=self.device).unsqueeze(0).repeat(states.shape[0], 1, 1)
        state_action_pairs = torch.cat([states_expanded, actions], dim=2)

        z_values = torch.stack([z_net(state_action_pairs) for z_net in self.z_networks])
        z_avg = torch.mean(z_values, dim=0)
        if len(self.z_networks) > 1:
            z_std = torch.std(z_values, dim=0)
        else:
            z_std = torch.zeros_like(z_avg)

        rewards = self.reward(state_action_pairs)
        
        # Scale rewards and normalize z_std
        rewards = self.reward_scale * rewards
        
        return rewards + z_avg + self.z_std_multiplier * z_std
    

    def update_policy(self, states, actions, eta):
        Q = self.compute_q_values(states)

        logits = self.policy(states)
        current_probs = torch.softmax(logits, dim=-1)

        # Policy gradient loss (removed KL divergence since we no longer have target network)
        loss = -torch.mean(torch.sum(current_probs * (eta * Q.squeeze(-1)), dim=1))

        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()

        return loss.item()


    def compute_z_std(self):
        with torch.no_grad():
            # Stack predictions from all Z networks
            z_values = torch.stack([z_net(self.fixed_sa_pairs) for z_net in self.z_networks])
            # Compute variance across Z networks (dim=0) for each state-action pair
            z_variance_per_sa = torch.std(z_values, dim=0)
            # Average variance across all state-action pairs
            avg_z_variance = z_variance_per_sa.mean().item()
        return avg_z_variance
