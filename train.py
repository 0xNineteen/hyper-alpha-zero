from board import Board
from network import PolicyValueNet
from mcts import MCTSPlayer
import numpy as np
import torch
import ray
from utils import get_augmented_data
import time
from mcts import Random
from tqdm import tqdm

# N actors with the 'best model' weights which self-play games and store data in replay buffer 
@ray.remote
class BestWeightsWorker:
    def __init__(self, board, replay_buffer, param_server, should_eval=False, n_playout=400):
        self.board = board
        self.policy = PolicyValueNet(board.width, board.height)
        self.player = MCTSPlayer(
            policy_value_fn=self.policy.policy_value_fn,
            is_self_play=True,
            n_playout=n_playout,
        )
        self.eval_player = MCTSPlayer(
            policy_value_fn=self.policy.policy_value_fn,
            is_self_play=False, # !! important
            n_playout=n_playout,
        )
        self.replay_buffer = replay_buffer
        self.param_server = param_server
        self.n_eval_games = 10
        self.should_eval = should_eval

        # # stats on m1 mac pro
        # self.opponent = Random() # wins 10/10 in 2 minutes 

        # 100it [01:44,  1.13s/it]0199) 
        # (BestWeightsWorker pid=70199) best: n wins vs opponent: 7 / 10
        self.opponent = MCTSPlayer(n_playout=400)

    def update_weights(self):
        weights = ray.get(self.param_server.get_weights.remote())
        self.policy.set_weights(weights)

    def run(self):
        i = 0
        if self.should_eval:
            pbar = tqdm()

        while True:
            i += 1
            self.update_weights() # todo: optimize with hash checking before pulling full statedict

            data = self.self_play_game()
            self.replay_buffer.add.remote(data)

            if self.should_eval:
                pbar.update(1)

            if i % 50 == 0 and self.should_eval: 
                wins = evaluate_n(self.board, self.eval_player, self.opponent, self.n_eval_games)
                print(f'best: n wins vs opponent: {wins} / {self.n_eval_games}')

    @torch.no_grad()
    def self_play_game(self):
        board, player = self.board, self.player

        board.init_board()
        data = []
        current_player = []
        while True:
            move, probs = player.get_action(board, with_probs=True)
            current_player.append(board.current_player)
            data.append((board.current_state(), probs))

            board.do_move(move)

            end, winner = board.game_end()
            if end:
                z = np.zeros(len(data))  # 0 = tie
                if winner:
                    z[np.array(current_player) == winner] = 1
                    z[np.array(current_player) != winner] = -1
                winner = z
                break

        state, probs = zip(*data)
        augmented_data = get_augmented_data(board, zip(state, probs, winner))
        return augmented_data

# 1 best model weights store 
@ray.remote 
class BestWeightsParameterServer():
    def __init__(self, weights) -> None:
        self.weights = weights
        self.write_lock = False

    def set_write_lock(self, lock):
        # cant lock when already locked
        if lock and self.write_lock:
            return False
        # unlocking or locking when unlocked = ok 
        self.write_lock = lock
        return True

    def get_write_lock(self):
        return self.write_lock
    
    def set_weights(self, weights):
        self.weights = weights
    
    def get_weights(self):
        return self.weights

#   - one distributed replay buffer 
# todo: update to ape-x style prioritized replay buffer
@ray.remote
class ReplayBuffer():
    def __init__(self, max_size=100000):
        self.max_size = max_size
        self.buffer = []
    
    def add(self, data):
        # print('rb add', len(data), f"({len(self.buffer)})")
        self.buffer.extend(data)
        if len(self.buffer) > self.max_size:
            self.buffer = self.buffer[-self.max_size:]
    
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        idxs = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in idxs]

#  M 'candidate models' which pull from the replay buffer and train 
# - each iteration they play against the 'best model' and if they win the 'best model' weights is updated 
@ray.remote
class CandidateWorker():
    def __init__(self, board, replay_buffer, param_server, n_train_steps, batch_size, n_eval_games=5, n_playout=400):
        self.board = board
        self.policy = PolicyValueNet(board.width, board.height)
        self.player = MCTSPlayer(
            policy_value_fn=self.policy.policy_value_fn,
            is_self_play=False,
            n_playout=n_playout,
        )
        self.replay_buffer = replay_buffer
        self.param_server = param_server

        self.best_policy = PolicyValueNet(board.width, board.height)
        self.best_player = MCTSPlayer(
            policy_value_fn=self.best_policy.policy_value_fn,
            is_self_play=False,
            n_playout=n_playout,
        )

        self.n_eval_games = n_eval_games
        self.batch_size = batch_size
        self.n_train_steps = n_train_steps

    def update_weights(self):
        weights = ray.get(self.param_server.get_weights.remote())
        self.policy.set_weights(weights)

    def run(self):
        self.update_weights()

        i = 0
        _train_steps = 0
        while True:
            # pull data from replay buffer + train
            rq_size = self.batch_size * min(i, self.n_train_steps)
            data = ray.get(self.replay_buffer.sample.remote(rq_size))
            if data is None: 
                time.sleep(1)
                continue
            i += 1

            for i in range(len(data) // self.batch_size):
                state, probs, winner = tuple(map(np.array, zip(*data[i * self.batch_size : (i + 1) * self.batch_size])))
                loss, entropy = self.policy.train_step(state, probs, winner)
                # print(f"loss: {loss}, entropy: {entropy}")
                _train_steps += 1

            # evaluate against best model
            if _train_steps >= self.n_train_steps: 
                # write lock the best weights
                result = ray.get(self.param_server.set_write_lock.remote(True))
                if not result: 
                    continue

                print("evaluating...") 
                self.evaluate()
                _train_steps = 0
                print("done evaluating...")

                self.param_server.set_write_lock.remote(False)

                self.update_weights() # if you cant beat em join em

    def evaluate(self):
        best_weights = ray.get(self.param_server.get_weights.remote())
        self.best_policy.set_weights(best_weights)

        wins = evaluate_n(self.board, self.player, self.best_player, self.n_eval_games)
        if wins > self.n_eval_games // 2: # > 50% win rate
            # todo: dont override if new weights have already been updated
            print(f"won {wins}/{self.n_eval_games}: updating best weights...")
            weights = self.policy.get_weights()
            self.param_server.set_weights.remote(weights)

@torch.no_grad()
def evaluate_n(board, player1, player2, n_games):
    wins = 0
    for _ in range(n_games):
        winner = evaluate(board, player1, player2)
        if winner == 1:
            wins += 1
    return wins

@torch.no_grad()
def evaluate(board, player1, player2):
    board.init_board()
    players = [player1, player2]
    i = 0
    while True:
        player = players[i % 2]
        move = player.get_action(board, reset_tree=True)
        board.do_move(move)
        i += 1
        end, winner = board.game_end()
        if end:
            break

    return winner


def train():
    ray.init()

    board = Board(width=6, height=6, n_in_row=3)
    # init weights
    policy = PolicyValueNet(board.width, board.height)


    batch_size = 256
    n_train_steps = 500
    n_eval_games = 10
    max_buffer_size = 1_000_000
    N_rollout_workers = 2
    M_trainer_workers = 4

    # train
    print("starting...")
    buffer = ReplayBuffer.remote(max_buffer_size)
    param_server = BestWeightsParameterServer.remote(policy.get_weights())

    N_rollout_workers -= 1
    rollout_workers = [BestWeightsWorker.remote(board, buffer, param_server) for _ in range(N_rollout_workers)]
    trainer_workers = [CandidateWorker.remote(board, buffer, param_server, n_train_steps, batch_size, n_eval_games) for _ in range(M_trainer_workers)]
    rollout_workers += [BestWeightsWorker.remote(board, buffer, param_server, True)]

    # start
    rs = [w.run.remote() for w in rollout_workers]
    ts = [w.run.remote() for w in trainer_workers]

    ray.get(rs + ts)

    # # self_play in parralel [x]
    # # train in parralel by reading from self_play [x]
    # # cython for mcts [x]

    # for iter in (pbar := tqdm(range(100))):
    #     # start 5 self-play games in parallel
    #     play_data = ray.get([self_play.remote(board, player) for _ in range(5)])
    #     for d in play_data:
    #         buffer.extend(d)

    #     # sample
    #     if len(buffer) < batch_size:
    #         continue

    #     for _ in range(n_train_steps):
    #         idxs = np.random.choice(len(buffer), batch_size, replace=False)
    #         state, probs, winner = zip(*[buffer[i] for i in idxs])
    #         loss, entropy = policy.train_step(state, probs, winner)
    #     pbar.set_description("loss: {}, entropy: {}".format(loss, entropy))

    #     # evaluate
    #     if (iter + 1) % eval_every == 0:
    #         print("evaluating...")
    #         winners = []
    #         for _ in tqdm(range(n_games_eval)):
    #             winner = evaluate(board, player, pure_mcts)
    #             winners.append(winner)
    #         n_alpha_wins = np.sum(winner == 1)
    #         print("alpha zero wins: {} / {}".format(n_alpha_wins, n_games_eval))


if __name__ == "__main__":
    train()
