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
    def __init__(self, state_dim, action_dim, num_of_NNs, learning_rate=1e-3, seed=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_of_NNs = num_of_NNs

         # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # Initialize neural networks
        self.policy = TwoLayerNet(state_dim, action_dim)
        self.reward = TwoLayerNet(state_dim + action_dim, 1)
        self.z_networks = [TwoLayerNet(state_dim + action_dim, 1) for _ in range(num_of_NNs)]
        
        # Initialize optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.reward_optimizer = optim.Adam(self.reward.parameters(), lr=learning_rate)
        self.z_optimizers = [optim.Adam(net.parameters(), lr=learning_rate) for net in self.z_networks]
        
    def select_action(self, state):
        state = torch.FloatTensor(state)
        logits = self.policy(state)
        action_probs = torch.softmax(logits, dim=-1)
        action = torch.multinomial(action_probs, 1).item()
        return action

    def update_cost(self, expert_states, expert_actions, policy_states, policy_actions, eta):
        expert_sa = self.encode_actions_concatenate_states(expert_states, expert_actions)
        policy_sa = self.encode_actions_concatenate_states(policy_states, policy_actions)
        
        expert_reward = self.reward(expert_sa).mean()
        policy_reward = self.reward(policy_sa).mean()
        
        # TODO: check if this is correct
        # we want to increase the expert reward, because we know the expert played well
        # we want to bring the policy reward closer to the expert reward
        # what stops me from increasing the expert reward and decresing the policy reward indefinitely? 
            # the fact that the policy is trained to maximize the reward. So if the reward is high for the expert, the policy will try to get the same reward
            # the policy reward is here just for reference
        loss = policy_reward - expert_reward # if we use reward. cost: expert_cost - policy_cost
        
        self.reward_optimizer.zero_grad()
        loss.backward()
        self.reward_optimizer.step()
        
        return loss.item()
    
    def update_z_at_index(self, states, actions, rewards, gamma, eta, z_index):
        sa = self.encode_actions_concatenate_states(states, actions)

        z_net = self.z_networks[z_index]
        z_opt = self.z_optimizers[z_index]
        
        z_values = z_net(sa).squeeze()
        
        # Calculate discounted future rewards for each state
        discounted_future_rewards = []
        for i in range(len(rewards)):
            future_rewards = rewards[i:]
            discounted = [gamma**j * r for j, r in enumerate(future_rewards)]
            discounted_future_rewards.append(sum(discounted))
        
        discounted_future_rewards = torch.FloatTensor(discounted_future_rewards)
        
        # Compute loss
        loss = torch.mean((z_values - discounted_future_rewards)**2)
        
        z_opt.zero_grad()
        loss.backward()
        z_opt.step()
    

    def update_policy(self, states, eta):
        states = torch.FloatTensor(states)
        
        # Compute Q-values
        states_expanded = states.unsqueeze(1).repeat(1, self.action_dim, 1)  # Shape: [batch_size, action_dim, state_dim]
        actions = torch.eye(self.action_dim).unsqueeze(0).repeat(states.shape[0], 1, 1)  # Shape: [batch_size, action_dim, action_dim]
        state_action_pairs = torch.cat([states_expanded, actions], dim=2) 

        z_values = torch.stack([z_net(state_action_pairs) for z_net in self.z_networks]) # TODO: check

        z_avg = torch.mean(z_values, dim=0)
        z_std = torch.std(z_values, dim=0)

        rewards = self.reward(state_action_pairs)
        
        # with the reward we want q to be an overestimation of the reward
        Q = rewards + z_avg + z_std  # Shape: [batch_size, action_dim]

        # Compute current policy probabilities
        logits = self.policy(states)
        current_probs = torch.softmax(logits, dim=-1)
        
        # Detach the current probabilities to represent the "old" policy
        old_probs = current_probs.detach()

        # Compute the loss
        # we add "-" because we want to maximize the reward and thus q
        loss = - torch.mean(torch.sum(current_probs * (eta * Q.squeeze(-1) + torch.log(current_probs) - torch.log(old_probs)), dim=1))

        # Update the policy
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()

        return loss.item()
    
    def encode_actions_concatenate_states(self, states, actions):
        actions_one_hot = torch.nn.functional.one_hot(torch.tensor(actions), num_classes=self.action_dim)
        return torch.vstack([torch.cat((torch.tensor(a1), a2.float())) for a1, a2 in zip(states, actions_one_hot)])