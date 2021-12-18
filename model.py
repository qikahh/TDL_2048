import math
import numpy as np
import torch 
from torch.nn import Sequential, Conv2d, Linear, Flatten, Dropout, Softmax, ReLU
from torch.optim import Adam
from torch.nn.modules import MSELoss
import os 

MAX_VALUE = 8192

class TDNet:
    def __init__(self,
                 n_actions,
                 n_features,
                 reward_decay=0.9,
                 ):
        self.n_actions = n_actions
        self.n_features = n_features

        self._init_tuples()

    def _init_tuples(self):
        self.tuple_list = [[(0,0)], [(0,0), (0,1)], [(0,1), (0,2)], [(0,1), (1,1)], [(1,1), (1,2)]]
        self.tuple_num = len(self.tuple_list)
        self.tuple_values = [[] for _ in range(self.tuple_num)]
        for i in range(self.tuple_num):
            self.tuple_values.append([MAX_VALUE for _ in range(5)])
        self.tuple_values = np.array(self.tuple_values)
    
    def tuple_to_state(self, tuple_sample):
        if len(tuple_sample) == 1:
            return int(tuple_sample[0] != 0)
        else:
            max_num = max(tuple_sample)
            min_num = min(tuple_sample)
            if min_num > 0:
                max_num = math.log(max_num, 2)
                min_num = math.log(min_num, 2)
                max_num -= (min_num-1)
                min_num = 1
                max_num = min(3, max_num)
                return 1+max_num
            else:
                return int(max_num != 0)

    def choose_action(self, state, det=False):
        tstate = state[np.newaxis, np.newaxis, :, :]
        tstate = torch.Tensor(self.preprocess_state(tstate)).to('cuda' if self.use_cuda else 'cpu')
        if det or np.random.uniform() < self.epsilon:
            action_value = self.q_eval_model(tstate)
            action_value = np.squeeze(action_value.detach().cpu().numpy())
            action_value = [action_value[i] if self.game.has_score(state.reshape([4, 4]), i) else np.min(action_value) - 10 for i in range(4)]
            action_index = np.argmax(action_value)
        else:
            action_value = [-1 if self.game.has_score(state.reshape([4, 4]), i) else np.random.random() for i in range(4)]
            action_index = np.argmax(action_value)
        return action_index

    def learn(self):
        pass

    def save_model(self, episode):
        pass

    def load_model(self, episode):
        pass