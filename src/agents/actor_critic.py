import gym
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Actor, self).__init__()
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


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class ActorCritic:
    def __init__(self, env):
        self.lr = 0.001
        self.gamma = 0.99

        self.state_dim = env.get_state_space()
        self.action_dim = env.get_action_space()
        self.action_bound_low, self.action_bound_high = env.get_action_bound()
        self.actor = Actor(self.state_dim, self.action_dim)
        self.critic = Critic(self.state_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        mu, std = self.actor(state)
        m = Normal(mu, std)
        action = m.sample()
        action = torch.tanh(action)
        action = action * (self.action_bound_high - self.action_bound_low) / 2.0 + (self.action_bound_high + self.action_bound_low) / 2.0
        return action.detach().numpy()[0], m.log_prob(action).sum(dim=-1)

    def learn(self, state, reward, next_state, done, log_prob):
        state = torch.FloatTensor(state).unsqueeze(0)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        reward = torch.FloatTensor([reward])
        done = torch.FloatTensor([done])
        
        value = self.critic(state)
        next_value = self.critic(next_state)
        td_target = reward + self.gamma * next_value * (1 - done)
        td_error = td_target - value

        critic_loss = (td_error ** 2).mean()
        actor_loss = -log_prob * td_error.detach()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def save_model(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        torch.save(self.actor.state_dict(), path)

    def load_model(self, path):
        self.actor.load_state_dict(torch.load(path))