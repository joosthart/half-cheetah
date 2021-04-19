import time
import sys

import mujoco_py
import gym
import numpy as np

env = gym.make('HalfCheetah-v2')

state = env.reset()
i = 0
while i < 1e3:
    env.render()
    action = env.action_space.sample()
    # action = np.array([0.5, 0.5, 0, 0, 0, 0])
    print('action: ', action)
    state = env.step(action)
    print('state: ', state)

    i += 1

input('Press any key to continue...')
env.close()

