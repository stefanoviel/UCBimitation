
import torch
import torch.optim as optim


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
    def __init__(self, state_dim, action_dim, num_of_NNs, learning_rate=1e-3, device='cpu', seed=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_of_NNs = num_of_NNs
        self.device = device

        if seed is not None:
            torch.manual_seed(seed)
            if str(device).startswith('cuda'):
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
            action = torch.multinomial(action_probs, 1)
        return action

    def update_cost(self, expert_states, expert_actions, policy_states, policy_actions, eta):
        expert_actions_one_hot = torch.nn.functional.one_hot(expert_actions, num_classes=self.action_dim)
        policy_actions_one_hot = torch.nn.functional.one_hot(policy_actions, num_classes=self.action_dim)
        
        # if we terminated there will be one more state than action
        expert_states = expert_states[:expert_actions_one_hot.shape[0]] 
        policy_states = policy_states[:policy_actions_one_hot.shape[0]] 

        expert_sa = torch.cat((expert_states, expert_actions_one_hot), dim=1)
        policy_sa = torch.cat((policy_states, policy_actions_one_hot), dim=1)
        
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
        actions_one_hot = torch.nn.functional.one_hot(actions, num_classes=self.action_dim)
        sa = torch.cat((states, actions_one_hot.float()), dim=1)

        z_net = self.z_networks[z_index]
        z_opt = self.z_optimizers[z_index]
        
        z_values = z_net(sa).squeeze()
        
        discounted_future_rewards = torch.zeros_like(rewards)
        running_sum = 0
        for t in reversed(range(len(rewards))):
            running_sum = rewards[t] + gamma * running_sum
            discounted_future_rewards[t] = running_sum
        
        loss = torch.mean((z_values - discounted_future_rewards)**2)
        
        z_opt.zero_grad()
        loss.backward()
        z_opt.step()
    
    def update_policy(self, states, eta):
        states_expanded = states.unsqueeze(1).repeat(1, self.action_dim, 1)
        actions = torch.eye(self.action_dim, device=self.device).unsqueeze(0).repeat(states.shape[0], 1, 1)
        state_action_pairs = torch.cat([states_expanded, actions], dim=2)

        z_values = torch.stack([z_net(state_action_pairs) for z_net in self.z_networks])

        z_avg = torch.mean(z_values, dim=0)
        z_std = torch.std(z_values, dim=0)

        c_values = self.cost(state_action_pairs)
        
        Q = c_values + z_avg + z_std

        logits = self.policy(states)
        current_probs = torch.softmax(logits, dim=-1)
        
        old_probs = current_probs.detach()

        loss = - torch.mean(torch.sum(current_probs * (eta * Q.squeeze(-1) + torch.log(current_probs) - torch.log(old_probs)), dim=1))

        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()

        return loss.item()

