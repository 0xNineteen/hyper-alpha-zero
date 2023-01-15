# -*- coding: utf-8 -*-
"""
@author: Junxiao Song
"""
import numpy as np


class Board:
    """board for the game"""

    def __init__(self, width=8, height=8, n_in_row=5):
        self.width = width
        self.height = height
        self.n_in_row = n_in_row

        # board states stored as a dict,
        # key: move as location on the board,
        # value: player as pieces type
        self.states = {}

        # need how many pieces in a row to win
        self.players = [1, 2]  # player1 and player2

    def init_board(self, start_player=0):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception(
                "board width and height can not be "
                "less than {}".format(self.n_in_row)
            )
        self.current_player = self.players[start_player]  # start player
        # keep available moves in a list
        self.availables = list(range(self.width * self.height))
        self.states = {}
        self.last_move = -1

    def do_move(self, move):
        self.states[move] = self.current_player
        self.availables.remove(move)
        self.current_player = self.players[
            (self.players.index(self.current_player) + 1) % 2
        ]
        self.last_move = move

    def move_to_location(self, move):
        """
        3*3 board's moves like:
        6 7 8
        3 4 5
        0 1 2
        and move 5's location is (1,2)
        """
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self):
        """return the board state from the perspective of the current player.
        state shape: 4*width*height
        4 channels:
            - player0 move
            - player1 move
            - last move
            - current player
        """
        square_state = np.zeros((4, self.width, self.height))

        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]

            square_state[0][move_curr // self.width, move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width, move_oppo % self.height] = 1.0

            # indicate the last move location
            square_state[2][
                self.last_move // self.width, self.last_move % self.height
            ] = 1.0

        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0  # indicate the colour to play

        return square_state[:, ::-1, :]

    def has_a_winner(self):
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = list(set(range(width * height)) - set(self.availables))
        if len(moved) < self.n_in_row * 2 - 1:
            return False, -1

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            if (
                w in range(width - n + 1)
                and len(set(states.get(i, -1) for i in range(m, m + n))) == 1
            ):
                return True, player

            if (
                h in range(height - n + 1)
                and len(set(states.get(i, -1) for i in range(m, m + n * width, width)))
                == 1
            ):
                return True, player

            if (
                w in range(width - n + 1)
                and h in range(height - n + 1)
                and len(
                    set(
                        states.get(i, -1)
                        for i in range(m, m + n * (width + 1), width + 1)
                    )
                )
                == 1
            ):
                return True, player

            if (
                w in range(n - 1, width)
                and h in range(height - n + 1)
                and len(
                    set(
                        states.get(i, -1)
                        for i in range(m, m + n * (width - 1), width - 1)
                    )
                )
                == 1
            ):
                return True, player

        return False, -1

    def game_end(self):
        """Check whether the game is ended or not"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player

    def show(self):
        """Draw the board and show game info"""
        width = self.width
        height = self.height
        player1, player2 = self.players

        print("Player", player1, "with X".rjust(3))
        print("Player", player2, "with O".rjust(3))
        print()
        for x in range(width):
            print("{0:8}".format(x), end="")
        print("\r\n")
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end="")
            for j in range(width):
                loc = i * width + j
                p = self.states.get(loc, -1)
                if p == player1:
                    print("X".center(8), end="")
                elif p == player2:
                    print("O".center(8), end="")
                else:
                    print("_".center(8), end="")
            print("\r\n\r\n")
