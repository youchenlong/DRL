import os
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import copy
from components.buffer import ReplayBuffer

class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DuelingQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc_value = nn.Linear(hidden_dim, 1)
        self.fc_advantage = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        value = self.fc_value(x)
        advantage = self.fc_advantage(x)
        q_values = value + (advantage - advantage.mean())
        return q_values

class DuelingDQN:
    def __init__(self, env):
        self.lr = 0.001
        self.gamma = 0.99
        self.epsilon = 0.01
        self.buffer_size = 50000
        self.batch_size = 64
        self.target_update_interval = 200
        self.idx = 0

        self.state_dim = env.get_state_space()
        self.action_dim = env.get_action_space()
        self.q_network = DuelingQNetwork(self.state_dim, self.action_dim)
        self.target_network = copy.deepcopy(self.q_network)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.buffer = ReplayBuffer(self.buffer_size, self.batch_size, self.state_dim, self.action_dim)

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state)
        return q_values.argmax().item()

    def learn(self):
        states, actions, rewards, next_states, dones = self.buffer.sample()

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.LongTensor(dones)

        current_Q = torch.gather(self.q_network(states), 1, actions.unsqueeze(1)).squeeze(1)
        target_Q = rewards + ((1 - dones) * self.gamma * torch.gather(self.target_network(next_states), 1, self.q_network(next_states).argmax(1).unsqueeze(1)).squeeze(1))
        td_loss = ((current_Q - target_Q) ** 2).mean()
        self.optimizer.zero_grad()
        td_loss.backward()
        self.optimizer.step()

        self.idx += 1
        if self.idx % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())        

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.store(state, action, reward, next_state, done)

    def save_model(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        torch.save(self.q_network.state_dict(), path)

    def load_model(self, path):
        self.q_network.load_state_dict(torch.load(path))