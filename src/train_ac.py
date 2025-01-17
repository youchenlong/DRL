import os
import numpy as np
import matplotlib.pyplot as plt
from envs import e_REGISTRY
from agents import a_REGISTRY
from utils.plot import *


def train(env_name="gym", map_name="CartPole-v1", alg_name="actor_critic_discrete"):
    env = e_REGISTRY[env_name](map_name)
    agent = a_REGISTRY[alg_name](env)

    num_episodes = 1005
    total_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        rewards = []
        done = False

        while not done:
            action, log_prob = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            rewards.append(reward)
            
            agent.learn(state, reward, next_state, done, log_prob)

            if done:
                total_reward = sum(rewards)
                print(f"Episode {episode + 1}, Total Reward: {total_reward}")
                break

            state = next_state

        total_rewards.append(total_reward)
        
        if episode % 500 == 0:
            agent.save_model("./results/models/{}/{}/{}/{}".format(alg_name, env_name, map_name, episode))
            save_data("./results/logs/{}/{}/{}".format(alg_name, env_name, map_name), "rewards.txt", total_rewards)
    env.close()


def evaluate(env_name="gym", map_name="CartPole-v1", alg_name="actor_critic_discrete"):
    env = e_REGISTRY[env_name](map_name)
    agent = a_REGISTRY[alg_name](env)

    if os.path.exists("./results/models/{}/{}/{}/{}".format(alg_name, env_name, map_name, 1000)):
        agent.load_model("./results/models/{}/{}/{}/{}".format(alg_name, env_name, map_name, 1000))

    num_episodes = 10

    for episode in range(num_episodes):
        state = env.reset()
        rewards = []
        done = False

        while not done:
            env.render()
            action, _ = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            rewards.append(reward)

            if done:
                print(f"Episode {episode + 1}, Total Reward: {sum(rewards)}")
                break

            state = next_state
    env.close()


if __name__ == '__main__':
    env_name = "gym"
    map_name = "MountainCarContinuous-v0"
    alg_name = "actor_critic"

    train(env_name, map_name, alg_name)

    # evaluate(env_name, map_name, alg_name)

    data = load_data("./results/logs/{}/{}/{}/rewards.txt".format(alg_name, env_name, map_name))
    plot(data)
