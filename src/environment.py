import numpy as np


def is_terminal(board):
    """
    If board is terminal, returns 1 or -1 depending on winner, or 0 if tie
    If board is not terminal, returns None
    """
    for turn in [-1, 1]:
        mask = board == turn
        out = mask.all(0).any() | mask.all(1).any()
        out |= np.diag(mask).all() | np.diag(mask[:, ::-1]).all()

        if out:
            return turn

    if not np.any(board == 0):
        return 0

    return None


class Environment:
    def __init__(self, size=3):
        self.board = np.zeros((size, size))
        self.size = size
        self.turn = 1

    def update(self, square):
        self.board[square] = self.turn
        self.turn *= -1

    def reset(self):
        self.board = np.zeros((self.size, self.size))
        self.turn = 1

    def board_to_state(self):
        n_states = 3 ** (self.size ** 2)
        state = 0

        for x in self.board.flatten():
            state += ((x + 1) / 3) * n_states
            n_states /= 3

        return int(state)

    def get_state(self):
        return {
            'state': self.board_to_state(),
            'board': self.board,
            'turn': self.turn,
            'reward': 0 if is_terminal(self.board) is None else is_terminal(self.board),
            'done': False if is_terminal(self.board) is None else True
        }

    def is_done(self):
        return False if is_terminal(self.board) is None else True
