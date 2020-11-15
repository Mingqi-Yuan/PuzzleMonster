from policy.DQN import DQNAgent
from simulator.NPuzzle import NPUZZLE
from utils import *

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import logging
logging.disable(30)
import tensorflow as tf
import cv2

GRID_X = 3
GRID_Y = 3
TOTAL_NUM = GRID_X * GRID_Y
N_STATE = 9
N_ACTION = 4
EPSILON = 1e-3
EPOCHS = 100
REPLAYER_CAPACITY = 1e+6
BATCH_SIZE = 256
BATCHES = 10
LR = 1e-3
LR_DECAY = 1e-5

if __name__ == '__main__':
    rl_env = NPUZZLE(
        grid_x=GRID_X,
        grid_y=GRID_Y
    )

    rl_agent = DQNAgent(
        n_state=N_STATE,
        n_action=N_ACTION,
        epsilon=EPSILON,
        lr=LR,
        lr_decay=LR_DECAY,
        batch_size=BATCH_SIZE,
        batches=BATCHES
    )

    rl_writer = tf.summary.create_file_writer(logdir='./log/train_rl/rl')

    for game in range(1):
        total_step = 0
        episode_reward = 0
        init_mat = np.array([[2, 6, 5],
                             [7, 3, 8],
                             [4, 0, 1]])

        print('Epoch={}, The Puzzle is solvable={}'.format(game, is_solvable(init_mat, rl_env.org_mat)))
        rl_state = rl_env.reset(init_mat)

        while True:
            rl_action = rl_agent.decide(rl_state)
            rl_next_state, rl_reward, rl_done = rl_env.step(rl_action)
            rl_agent.learn(rl_state, rl_action, rl_reward, rl_next_state, rl_done)
            print('Epoch={}, Step={}, RL action={}, Reward={}'.format(game, total_step, rl_action, rl_reward))

            episode_reward += rl_reward
            total_step += 1

            ''' visualize '''
            visual_mat = fig2data(rl_env.cur_mat)
            cv2.imshow('NPUZZLE', visual_mat)
            cv2.waitKey(1)

            if rl_done:
                cv2.destroyAllWindows()
                break

        with rl_writer.as_default():
            tf.summary.scalar('Episode reward', episode_reward, step=game)
            tf.summary.scalar('Total steps', total_step, step=game)
