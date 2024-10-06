import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

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
    def __init__(self, state_dim, action_dim, num_of_NNs, learning_rate=1e-3, device='cpu', seed=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_of_NNs = num_of_NNs
        self.device = device

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if device == 'cuda':
                torch.cuda.manual_seed(seed)
        
        self.policy = TwoLayerNet(state_dim, action_dim).to(device)
        self.cost = TwoLayerNet(state_dim + action_dim, 1).to(device)
        self.z_networks = [TwoLayerNet(state_dim + action_dim, 1).to(device) for _ in range(num_of_NNs)]
        
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.cost_optimizer = optim.Adam(self.cost.parameters(), lr=learning_rate)
        self.z_optimizers = [optim.Adam(net.parameters(), lr=learning_rate) for net in self.z_networks]
        
    def select_action(self, state):
        with torch.no_grad():
            logits = self.policy(state)
            action_probs = torch.softmax(logits, dim=-1)
            action = torch.multinomial(action_probs, 1).item()
        return action

    def update_cost(self, expert_states, expert_actions, policy_states, policy_actions, eta):
        expert_actions_one_hot = torch.nn.functional.one_hot(torch.tensor(expert_actions, device=self.device), num_classes=self.action_dim)
        policy_actions_one_hot = torch.nn.functional.one_hot(torch.tensor(policy_actions, device=self.device), num_classes=self.action_dim)
        
        expert_sa = torch.vstack([torch.cat((torch.tensor(a1, device=self.device), a2.float())) for a1, a2 in zip(expert_states, expert_actions_one_hot)])
        policy_sa = torch.vstack([torch.cat((torch.tensor(a1, device=self.device), a2.float())) for a1, a2 in zip(policy_states, policy_actions_one_hot)])
        
        expert_cost = self.cost(expert_sa).mean()
        policy_cost = self.cost(policy_sa).mean()

        # TODO: check if this is correct
        # we want to increase the expert reward, because we know the expert played well
        # we want to bring the policy reward closer to the expert reward
        # what stops me from increasing the expert reward and decresing the policy reward indefinitely? 
            # the fact that the policy is trained to maximize the reward. So if the reward is high for the expert, the policy will try to get the same reward
            # the policy reward is here just for reference
        loss = expert_cost - policy_cost
        self.cost_optimizer.zero_grad()
        loss.backward()
        self.cost_optimizer.step()
        
        return loss.item()
    
    def update_z_at_index(self, states, actions, rewards, gamma, eta, z_index):
        actions_one_hot = torch.nn.functional.one_hot(torch.tensor(actions, device=self.device), num_classes=self.action_dim)
        sa = torch.vstack([torch.cat((torch.tensor(a1, device=self.device), a2.float())) for a1, a2 in zip(states, actions_one_hot)])

        z_net = self.z_networks[z_index]
        z_opt = self.z_optimizers[z_index]
        
        z_values = z_net(sa).squeeze()
        
        discounted_future_rewards = []
        for i in range(len(rewards)):
            future_rewards = rewards[i:]
            discounted = [gamma**j * r for j, r in enumerate(future_rewards)]
            discounted_future_rewards.append(sum(discounted))
        
        discounted_future_rewards = torch.FloatTensor(discounted_future_rewards).to(self.device)
        
        loss = torch.mean((z_values - discounted_future_rewards)**2)
        
        z_opt.zero_grad()
        loss.backward()
        z_opt.step()
    
    def update_policy(self, states, eta):
        states = torch.FloatTensor(states).to(self.device)
        
        states_expanded = states.unsqueeze(1).repeat(1, self.action_dim, 1)  # Shape: [batch_size, action_dim, state_dim]
        actions = torch.eye(self.action_dim, device=self.device).unsqueeze(0).repeat(states.shape[0], 1, 1) # Shape: [batch_size, action_dim, action_dim]
        state_action_pairs = torch.cat([states_expanded, actions], dim=2) 

        z_values = torch.stack([z_net(state_action_pairs) for z_net in self.z_networks])

        z_avg = torch.mean(z_values, dim=0)
        z_std = torch.std(z_values, dim=0)

        c_values = self.cost(state_action_pairs)
        
        Q = c_values + z_avg + z_std # Shape: [batch_size, action_dim]

        logits = self.policy(states)
        current_probs = torch.softmax(logits, dim=-1)
        
        old_probs = current_probs.detach()

        # Compute the loss
        # we add "-" because we want to maximize the reward and thus q => minimize the loss with a minus in front
        loss = - torch.mean(torch.sum(current_probs * (eta * Q.squeeze(-1) + torch.log(current_probs) - torch.log(old_probs)), dim=1))

        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()

        return loss.item()