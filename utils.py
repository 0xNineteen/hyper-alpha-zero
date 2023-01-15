import numpy as np
import torch

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

def get_augmented_data(board, play_data):
    """augment the data set by rotation and flipping
    play_data: [(state, mcts_prob, winner_z), ..., ...]
    """
    extend_data = []
    for state, probs, winner in play_data:
        for i in [1, 2, 3, 4]:
            # rotate counterclockwise
            equi_state = np.array([np.rot90(s, i) for s in state])
            equi_mcts_prob = np.rot90(
                np.flipud(probs.reshape(board.height, board.width)), i
            )
            extend_data.append(
                (equi_state, np.flipud(equi_mcts_prob).flatten(), winner)
            )
            # flip horizontally
            equi_state = np.array([np.fliplr(s) for s in equi_state])
            equi_mcts_prob = np.fliplr(equi_mcts_prob)
            extend_data.append(
                (equi_state, np.flipud(equi_mcts_prob).flatten(), winner)
            )

    return extend_data
