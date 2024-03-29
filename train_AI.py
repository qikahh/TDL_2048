import torch
from game_2048 import Game2048
import numpy as np
from model import DeepQNetwork

score_list = []


def run_2048(load_episode = None):
    step = 0
    if load_episode != None:
        RL.load_model(load_episode)
    for episode in range(5000):
        print('episode:', episode)
        game.reset()
        s = game.get_state()
        game_step = 0
        while True:
            action_index = RL.choose_action(s)
            s_, r, done = game.step(action_index)
            # print('action:', game.actions[action_index])
            # print('game:\n', s_, '\n')
            RL.store_memory(s.reshape([-1, ]), s_.reshape([-1, ]), action_index, r)
            if step > 100 and step % 5 == 0:
                RL.learn()

            s = s_
            if done:
                print('game:\n', game.board)
                print('max score:', game.get_score(), ' board sum:', np.sum(game.board), ' play step:', game.n_step)
                score_list.append(game.get_score())
                break
            step += 1
            game_step += 1

        if episode > 200 and (episode + 1) % 50 == 0:
            RL.save_model(episode + 1)
            print('model saved!')
    print('game over')
    RL.save_model(episode + 1)
    print('model saved!')


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    game = Game2048()
    
    RL = DeepQNetwork(n_actions=game.n_actions,
                      n_features=game.n_features,
                      gameref=game,
                      memory_size=10000,
                      batch_size=128,
                      train_epochs=20,
                      use_cuda=use_cuda,
                      modeldir='model',
                      logdir='log')
    run_2048()
    print(score_list)
    np.savetxt('score_list.txt', np.array(score_list))
