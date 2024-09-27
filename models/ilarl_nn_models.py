import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle


# Neural Network architecture for policy, cost function, and value function estimators
class TwoLayerNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TwoLayerNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ImitationLearning:
    def __init__(self, state_dim, action_dim, num_of_NNs, learning_rate=1e-3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_of_NNs = num_of_NNs
        
        # Initialize neural networks
        self.policy = TwoLayerNet(state_dim, action_dim)
        self.cost = TwoLayerNet(state_dim + action_dim, 1)
        self.z_networks = [TwoLayerNet(state_dim + action_dim, 1) for _ in range(num_of_NNs)]
        
        # Initialize optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.cost_optimizer = optim.Adam(self.cost.parameters(), lr=learning_rate)
        self.z_optimizers = [optim.Adam(net.parameters(), lr=learning_rate) for net in self.z_networks]
        
    def select_action(self, state):
        state = torch.FloatTensor(state)
        logits = self.policy(state)
        action_probs = torch.softmax(logits, dim=-1)
        action = torch.multinomial(action_probs, 1).item()
        return action
    
    def update_cost(self, expert_states, expert_actions, policy_states, policy_actions, eta):
        expert_sa = torch.cat([torch.FloatTensor(expert_states), torch.FloatTensor(expert_actions)], dim=1)
        policy_sa = torch.cat([torch.FloatTensor(policy_states), torch.FloatTensor(policy_actions)], dim=1)
        
        expert_cost = self.cost(expert_sa).mean()
        policy_cost = self.cost(policy_sa).mean()
        
        loss = expert_cost - policy_cost
        
        self.cost_optimizer.zero_grad()
        loss.backward()
        self.cost_optimizer.step()
        
        return loss.item()
    
    def update_z(self, states, actions, rewards, gamma, eta):
        sa = torch.cat([torch.FloatTensor(states), torch.FloatTensor(actions)], dim=1)
        discounted_rewards = torch.FloatTensor([gamma**i * r for i, r in enumerate(rewards)])
        
        for i, (z_net, z_opt) in enumerate(zip(self.x, self.z_optimizers)):
            z_values = z_net(sa).squeeze()
            target = 2 * gamma * torch.roll(discounted_rewards, -1)[:-1]
            loss = torch.mean((z_values[:-1] - target)**2)
            
            z_opt.zero_grad()
            loss.backward()
            z_opt.step()
    
    
    def update_policy(self, states, eta):
        states = torch.FloatTensor(states)
        
        # Compute Q-values
        z_values = torch.stack([z_net(torch.cat([states, torch.eye(self.action_dim).repeat(states.shape[0], 1, 1)], dim=1))
                                for z_net in self.z_networks])
        z_avg = torch.mean(z_values, dim=0)
        z_std = torch.std(z_values, dim=0)
        
        c_values = self.cost(torch.cat([states, torch.eye(self.action_dim).repeat(states.shape[0], 1, 1)], dim=1))
        
        Q = c_values + z_avg - z_std  # Shape: [batch_size, action_dim]

        # Compute current policy probabilities
        logits = self.policy(states)
        current_probs = torch.softmax(logits, dim=-1)
        
        # Detach the current probabilities to represent the "old" policy
        old_probs = current_probs.detach()

        # Compute the loss
        loss = -torch.mean(torch.sum(current_probs * (eta * Q + torch.log(current_probs) - torch.log(old_probs)), dim=1))

        # Update the policy
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()

        return loss.item()