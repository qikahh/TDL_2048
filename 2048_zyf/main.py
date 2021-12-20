import time
import numpy as np
from os.path import join
import pickle
import random

from World import Board, showstatus
from Agent import Agent, load_model

EPOCH_SIZE = 50000
MAX_NUM = 12 # 1<<12 == 4096
ACTION_NUM = 4 # up, down, left, right
NAME = '6_tuple'
squeeze = False
merge = True
if squeeze:
    NAME += '_squeeze'
if merge:
    NAME += '_merge'

PATTERNS =  [
    [0,1,2],
    [4,5,6],
    [3,6,7],
    [7,10,11],
    [2,3,7]
]

MERGE_PATTERNS = [
    [0,1],
    [0,2],
    [1,3],
    [1,4]
]

if __name__ == '__main__':
    random.seed(2021)
    path = join('models', NAME)
    agent = Agent(NAME, PATTERNS, MERGE_PATTERNS, MAX_NUM, merge, squeeze)
    # agent = load_model(path)
    '''
    game = Board().popup().popup()
    print(game.__str__())
    while game.end() == False:
        next_game, reward, action = agent.play(game)
        next_game = next_game.popup()
        print('reward:', reward)
        print('action:', action)
        print(next_game.__str__())
        game = next_game
    '''
    agent.train(EPOCH_SIZE)