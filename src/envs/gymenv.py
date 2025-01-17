import gym

class GymEnvironment:
    def __init__(self, map_name):
        self.map_name = map_name
        self.env = gym.make(map_name)

    def reset(self):
        self.state = self.env.reset()
        return self.state

    def step(self, action):
        self.state, reward, done, info = self.env.step(action)
        return self.state, reward, done, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
    
    def get_state_space(self):
        return self.env.observation_space.shape[0]
    
    def get_action_space(self):
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            return self.env.action_space.n
        elif isinstance(self.env.action_space, gym.spaces.Box):
            return self.env.action_space.shape[0]
        else:
            raise ValueError("Unsupported action space type")

    def get_action_bound(self):
        if isinstance(self.env.action_space, gym.spaces.Box):
            return self.env.action_space.low[0], self.env.action_space.high[0]
        else:
            raise ValueError("Unsupported action space type")
    
    def sample_action(self):
        return self.env.action_space.sample()


if __name__ == "__main__":
    env = GymEnvironment('Pendulum-v1') 
    env.reset()
    done = False
    while not done:
        action = env.sample_action()  
        state, reward, done, info = env.step(action)
        if done:
            env.reset()
    env.close()