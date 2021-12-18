
from game_2048 import Game2048
import numpy as np

from model import DeepQNetwork
import time
import torch

game = Game2048()
game.reset()
s = game.get_state()
game_step = 0
use_cuda = True
device = 'cuda' if use_cuda else 'cpu'
model = DeepQNetwork(game.actions, game.n_features, use_cuda=use_cuda, modeldir='.')
#model.load_keras_model(load_model('pretrained/dqn2048_cnn.h5'))
model.load_model(0)
print(game.board)
print()
while True:
    state = s[np.newaxis, np.newaxis, :, :]
    state = np.log(state + 1) / 16
    action_index = model.q_eval_model(torch.Tensor(state).to(device)).detach().cpu().numpy().squeeze()
    action_index = [action_index[i] if game.has_score(game.board, i) else np.min(action_index) - 10 for i in range(4)]
    choosed = np.argmax(action_index)
    s_, r, done = game.step(choosed)
    print('action:', game.actions[choosed])
    print('game:\n', s_, '\n')

    s = s_
    if done:
        print('final:\n', game.board)
        print('score:', game.get_score(), ' board sum:', np.sum(game.board), ' play step:', game.n_step)
        break

    game_step += 1
    time.sleep(1)
