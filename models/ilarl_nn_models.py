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
        x = torch.relu(self.fc1(x.float()))
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
        # One-hot encode the actions
        expert_actions_one_hot = torch.nn.functional.one_hot(torch.tensor(expert_actions), num_classes=self.action_dim)
        policy_actions_one_hot = torch.nn.functional.one_hot(torch.tensor(policy_actions), num_classes=self.action_dim)
        
        # Concatenate states and one-hot encoded actions
        expert_sa = torch.vstack([torch.cat((torch.tensor(a1), a2.float())) for a1, a2 in zip(expert_states, expert_actions_one_hot)])
        policy_sa = torch.vstack([torch.cat((torch.tensor(a1), a2.float())) for a1, a2 in zip(policy_states, policy_actions_one_hot)])
        
        expert_cost = self.cost(expert_sa).mean()
        policy_cost = self.cost(policy_sa).mean()
        
        # TODO: then we assume that the expert is optimal? becuase if we get something better than the expert it will have a lower lowwer loss
        loss = expert_cost - policy_cost
        
        self.cost_optimizer.zero_grad()
        loss.backward()
        self.cost_optimizer.step()
        
        return loss.item()
    
    def update_z_at_index(self, states, actions, rewards, gamma, eta, z_index):

        actions_one_hot = torch.nn.functional.one_hot(torch.tensor(actions), num_classes=self.action_dim)
        sa = torch.vstack([torch.cat((torch.tensor(a1), a2.float())) for a1, a2 in zip(states, actions_one_hot)])

        discounted_rewards = torch.FloatTensor([gamma**i * r for i, r in enumerate(rewards)])
        
        z_net = self.z_networks[z_index]
        z_opt = self.z_optimizers[z_index]
        
        z_values = z_net(sa).squeeze()
        target = gamma * torch.roll(discounted_rewards, -1)[:-1]
        loss = torch.mean((z_values[:-1] - target)**2)
        
        z_opt.zero_grad()
        loss.backward()
        z_opt.step()
    

    def update_policy(self, states, eta):
        states = torch.FloatTensor(states)
        
        # Compute Q-values
        states_expanded = states.unsqueeze(1).repeat(1, self.action_dim, 1)  # Shape: [batch_size, action_dim, state_dim]
        actions = torch.eye(self.action_dim).unsqueeze(0).repeat(states.shape[0], 1, 1)  # Shape: [batch_size, action_dim, action_dim]
        state_action_pairs = torch.cat([states_expanded, actions], dim=2) 

        z_values = torch.stack([z_net(state_action_pairs)  # Concatenate along the last dimension
                                for z_net in self.z_networks])

        z_avg = torch.mean(z_values, dim=0)
        z_std = torch.std(z_values, dim=0)

        c_values = self.cost(state_action_pairs)
        
        Q = c_values + z_avg - z_std  # Shape: [batch_size, action_dim]

        # Compute current policy probabilities
        logits = self.policy(states)
        current_probs = torch.softmax(logits, dim=-1)
        
        # Detach the current probabilities to represent the "old" policy
        old_probs = current_probs.detach()

        # Compute the loss
        loss = -torch.mean(torch.sum(current_probs * (eta * Q.squeeze(-1) + torch.log(current_probs) - torch.log(old_probs)), dim=1))

        # Update the policy
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()

        return loss.item()