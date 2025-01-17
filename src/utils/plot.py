import os
import numpy as np
import matplotlib.pyplot as plt
from envs import e_REGISTRY
from agents import a_REGISTRY


def save_data(dirname, filename, data):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(os.path.join(dirname, filename), 'w') as f:
        f.write(str(data))


def load_data(path):
    with open(path, 'r') as f:
        data = eval(f.read())
    return np.array(data)


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


def plot(data):
    window_sizes = [10, 50, 100]
    plt.figure(figsize=(10, 6))
    # plt.plot(data)
    for window_size in window_sizes:
        smoothed_data = moving_average(data, window_size)
        plt.plot(np.arange(window_size-1, len(data)), smoothed_data, label=f'Window size = {window_size}')
    plt.title('Episode Reward')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.legend()
    plt.show()
