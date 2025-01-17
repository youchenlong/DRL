# Deep Reinforcement Learning

## Agent

* Value-based
    - [ ] DQN
    - [ ] Double DQN
    - [ ] Dueling DQN
    - [ ] PER DQN

* Policy-based
    - [x] REINFORCE
    - [x] Actor-Critic
    - [x] DDPG

## Environment
- [x] Gym
* classic control 

    | env_name | state | action |
    | - | - | - |
    | CartPole-v1 | (4,) | Discrete(2) |
    | MountainCar-v0 | (2,) | Discrete(3) |
    | MountainCarContinuous-v0 | (2,) | Box(-1.0, 1.0, (1,), float32) |
    | Pendulum-v1 | (3,) | Box(-2.0, 2.0, (1,), float32) |

* Box2D

    | env_name | state | action |
    | - | - | - |
    | BipedalWalker-v3 | (24,) | Box(-1.0, 1.0, (4,), float32) |
    | LunarLander-v2 | (8,) | Discrete(4) | 

* Mujuco

    | env_name | state | action |
    | - | - | - |
    | Ant-v4 | (27,) | Box(-1.0, 1.0, (8,), float32) |
    | HalfCheetah-v4 | (17,) | Box(-1.0, 1.0, (6,), float32) |
    | Hopper-v4 | (11,) | Box(-1.0, 1.0, (3,), float32) |
    | HumanoidStandup-v4 | (376,) | Box(-0.4, 0.4, (17,), float32) |
    | Humanoid-v4 | (376,) | Box(-0.4, 0.4, (17,), float32) |
    | InvertedDoublePendulum-v4 | (11,) | Box(-1.0, 1.0, (1,), float32) |
    | InvertedPendulum-v4 | (4,) | Box(-3.0, 3.0, (1,), float32) |
    | Reacher-v4 | (11,) | Box(-1.0, 1.0, (2,), float32) |
    | Swimmer-v4 | (8,) | Box(-1.0, 1.0, (2,), float32) |
    | Walker2d-v4 | (17,) | Box(-1.0, 1.0, (6,), float32) |