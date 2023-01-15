# -*- coding: utf-8 -*-
"""
A pure implementation of the Monte Carlo Tree Search (MCTS)

@author: Junxiao Song
@author: x19
"""
import numpy as np
import copy

# from node import TreeNode # pure python
from ctree.cmcts import Node  # cython


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


def policy_value_fn(board):
    """a function that takes in a state and outputs a list of (action, probability)
    tuples and a score for the state"""
    # return uniform probabilities and 0 score for pure MCTS
    action_probs = np.ones(len(board.availables)) / len(board.availables)
    return (board.availables, action_probs), 0


class MCTS:
    """A simple implementation of Monte Carlo Tree Search."""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = Node(None, 1.0, 0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root
        while 1:
            if node.is_leaf():
                break
            # Greedily select next move.
            node, action = node.select(self._c_puct)
            state.do_move(action)

        (action, probs), leaf_value = self._policy(state)

        # Check for end of game
        end, winner = state.game_end()
        if not end:
            node.expand(action, probs)
        else:
            if winner == -1:
                leaf_value = 0.0
            else:
                leaf_value = 1.0 if winner == state.get_current_player() else -1.0

        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)

    def get_move(self, state, temp=1e-3):
        """Runs all playouts sequentially and returns the most visited action.
        state: the current game state

        Return: the selected action
        """
        for _ in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        # calc the move probabilities based on visit counts at the root node
        act_visits = [(node.action, node.n_visits) for node in self._root.children]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        child = self._root.get_child_by_action(last_move)
        if child:
            del self._root
            self._root = child
        else:
            print("WARNING: no child found for action", last_move)
            self.reset()

    def reset(self):
        self._root.delete_tree()  # del root + children
        self._root = Node(None, 1.0, 0)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    """AI player based on MCTS"""

    def __init__(
        self,
        policy_value_fn=policy_value_fn,
        c_puct=5,
        n_playout=2000,
        add_noise=True,
        is_self_play=False,
    ):
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)
        self.add_noise = add_noise
        self.is_self_play = is_self_play

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.reset()

    def get_action(self, board, with_probs=False, reset_tree=False):
        sensible_moves = board.availables

        if len(sensible_moves) > 0:
            acts, act_probs = self.mcts.get_move(board)

            # mask out illegal moves
            full_probs = np.zeros(board.width * board.height)
            full_probs[list(acts)] = act_probs

            if self.add_noise:
                # explore relative to how many moves left
                game_left = len(board.availables) / (board.width * board.height)
                move = np.random.choice(
                    acts,
                    p=(1 - game_left) * act_probs
                    + game_left * np.random.dirichlet(0.3 * np.ones(len(act_probs))),
                )
            else:
                move = np.random.choice(acts, p=act_probs)

            if self.is_self_play and not reset_tree:
                # next turn will re-use the tree as the opponent
                self.mcts.update_with_move(move)
            else:
                self.mcts.reset()

            return move if not with_probs else (move, full_probs)
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "MCTS {}".format(self.player)


class Random:
    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board, reset_tree=False):
        return np.random.choice(board.availables)


if __name__ == "__main__":
    from board import Board

    board = Board()
    board.init_board()

    # mcts = MCTS(policy_value_fn, n_playout=1_000)
    # acts, probs = mcts.get_move(board)
    # move = acts[0]
    # mcts.update_with_move(move)
    # # for _ in range(100000):
    # #     board_copy = copy.deepcopy(board)
    # #     mcts._playout(board_copy)

    mcts_player = MCTSPlayer()
    mcts_player.set_player_ind(1)

    rando = Random()
    rando.set_player_ind(2)

    players = [mcts_player, rando]
    name = {1: "MCTS", 2: "Random"}

    i = 0
    while True:
        player = players[i % 2]
        print("player: ", name[player.player])
        move = player.get_action(board)
        board.do_move(move)
        i += 1
        end, winner = board.game_end()
        if end:
            break

    print("WINNER: ", name[winner], "Player: ", winner)
    board.show()
