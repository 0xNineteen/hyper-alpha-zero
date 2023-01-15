# -*- coding: utf-8 -*-
"""
An implementation of the policyValueNet in PyTorch
Tested in PyTorch 0.2.0 and 0.3.0

@author: Junxiao Song
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# todo: replace with jax + support tpus
class Net(nn.Module):
    """policy-value network module"""

    def __init__(self, board_width, board_height):
        super(Net, self).__init__()

        self.board_width = board_width
        self.board_height = board_height
        self.n = board_height * board_width

        # common layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # action policy layers
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4 * self.n, self.n)

        # state value layers
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2 * self.n, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, state_input):
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # action policy layers
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4 * self.board_width * self.board_height)
        x_act = F.log_softmax(self.act_fc1(x_act), dim=-1)

        # state value layers
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2 * self.board_width * self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = torch.tanh(self.val_fc2(x_val))

        return x_act, x_val


class PolicyValueNet:
    """policy-value network"""

    def __init__(self, board_width, board_height, lr=1e-3):
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4  # coef of l2 penalty

        # the policy value net module
        self.policy_value_net = Net(board_width, board_height)
        self.optimizer = optim.Adam(
            self.policy_value_net.parameters(), lr, weight_decay=self.l2_const
        )

    def get_weights(self):
        return self.policy_value_net.state_dict()

    def set_weights(self, sd):
        self.policy_value_net.load_state_dict(sd)

    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        state_batch = torch.tensor(state_batch).float()
        log_act_probs, value = self.policy_value_net(state_batch)
        act_probs = np.exp(log_act_probs.data.numpy())
        return act_probs, value.data.numpy()

    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        legal_positions = board.availables
        current_state = np.ascontiguousarray(
            board.current_state().reshape(-1, 4, self.board_width, self.board_height)
        )

        log_act_probs, value = self.policy_value_net(
            torch.from_numpy(current_state).float()
        )
        act_probs = np.exp(log_act_probs.data.numpy().flatten())

        act_probs = (legal_positions, act_probs[legal_positions])
        value = value.flatten()[0]

        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch):
        """perform a training step"""
        state_batch = torch.tensor(state_batch).float()
        mcts_probs = torch.tensor(mcts_probs).float()
        winner_batch = torch.tensor(winner_batch).float()

        # forward
        log_act_probs, value = self.policy_value_net(state_batch)

        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # Note: the L2 penalty is incorporated in optimizer
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, 1))
        loss = value_loss + policy_loss

        # backward and optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # calc policy entropy, for monitoring only
        entropy = -torch.mean(torch.sum(torch.exp(log_act_probs) * log_act_probs, 1))

        return loss.item(), entropy.item()


if __name__ == "__main__":
    from board import Board

    board = Board()
    net = Net(board.width, board.height)
    policy = PolicyValueNet(board.width, board.height)
    board.init_board()

    from mcts import MCTSPlayer

    mcts_player = MCTSPlayer(policy_value_fn=policy.policy_value_fn)
    mcts_player.set_player_ind(1)

    move = mcts_player.get_action(board)
    board.do_move(move)

    board.show()
