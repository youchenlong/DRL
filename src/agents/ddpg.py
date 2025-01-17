import gym
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import copy
from components.buffer import ReplayBuffer
from components.noise import OUNoise


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class DDPG:
    def __init__(self, env):
        self.lr = 0.001
        self.gamma = 0.99
        self.tau = 0.005
        self.buffer_size = 50000
        self.batch_size = 64
        self.noise_scale = 0.1

        self.state_dim = env.get_state_space()
        self.action_dim = env.get_action_space()
        self.action_bound_low, self.action_bound_high = env.get_action_bound()

        self.actor = Actor(self.state_dim, self.action_dim)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)

        self.critic = Critic(self.state_dim, self.action_dim)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)

        self.buffer = ReplayBuffer(self.buffer_size, self.batch_size, self.state_dim, self.action_dim)
        self.noise = OUNoise(self.action_dim)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().cpu().numpy().flatten()
        action = action * (self.action_bound_high - self.action_bound_low) / 2.0 + (self.action_bound_high + self.action_bound_low) / 2.0
        noise = self.noise.sample() * self.noise_scale
        action = np.clip(action + noise, self.action_bound_low, self.action_bound_high)
        return action

    def learn(self):
        states, actions, rewards, next_states, dones = self.buffer.sample()

        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        next_actions = self.actor_target(next_states)
        target_Q = self.critic_target(next_states, next_actions)
        target_Q = rewards + ((1 - dones) * self.gamma * target_Q).detach()
        current_Q = self.critic(states, actions)
        critic_loss = ((current_Q - target_Q) ** 2).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data = self.tau * param.data + (1 - self.tau) * target_param.data

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data = self.tau * param.data + (1 - self.tau) * target_param.data

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.store(state, action, reward, next_state, done)

    def save_model(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.actor.state_dict(), f"{path}_actor.pth")
        torch.save(self.critic.state_dict(), f"{path}_critic.pth")

    def load_model(self, path):
        self.actor.load_state_dict(torch.load(f"{path}_actor.pth"))
        self.critic.load_state_dict(torch.load(f"{path}_critic.pth"))
