import os
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, action_dim)
        self.fc_log_std = nn.Linear(hidden_dim, action_dim)
        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        mu = torch.tanh(self.fc_mu(x))
        log_std = self.fc_log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return mu, std

class REINFORCE:
    def __init__(self, env):
        self.lr = 0.001
        self.gamma = 0.99

        self.state_dim = env.get_state_space()
        self.action_dim = env.get_action_space()
        self.action_bound_low, self.action_bound_high = env.get_action_bound()
        self.policy_net = PolicyNetwork(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        mu, std = self.policy_net(state)
        m = Normal(mu, std)
        action = m.sample()
        action = torch.tanh(action)
        action = action * (self.action_bound_high - self.action_bound_low) / 2.0 + (self.action_bound_high + self.action_bound_low) / 2.0
        return action.detach().numpy()[0], m.log_prob(action).sum(dim=-1)

    def learn(self, rewards, log_probs):
        discounted_rewards = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            discounted_rewards.insert(0, R)
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        log_probs = torch.stack(log_probs)
        
        loss = -torch.sum(log_probs * discounted_rewards)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path))