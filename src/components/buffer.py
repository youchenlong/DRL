import numpy as np
from collections import deque


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, state_dim, action_dim):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.state = deque(maxlen=buffer_size)
        self.action = deque(maxlen=buffer_size)
        self.reward = deque(maxlen=buffer_size)
        self.next_state = deque(maxlen=buffer_size)
        self.done = deque(maxlen=buffer_size)

    def can_sample(self):
        return self.batch_size <= len(self.state)

    def sample(self):
        if self.can_sample():
            indices = np.random.choice(len(self.state), self.batch_size, replace=False)
            batch_state = [self.state[i] for i in indices]
            batch_action = [self.action[i] for i in indices]
            batch_reward = [self.reward[i] for i in indices]
            batch_next_state = [self.next_state[i] for i in indices]
            batch_done = [self.done[i] for i in indices]

            return np.array(batch_state), \
                    np.array(batch_action), \
                    np.array(batch_reward), \
                    np.array(batch_next_state), \
                    np.array(batch_done)
        return None

    def store(self, state, action, reward, next_state, done):
        self.state.append(state)
        self.action.append(action)
        self.reward.append(reward)
        self.next_state.append(next_state)
        self.done.append(done)