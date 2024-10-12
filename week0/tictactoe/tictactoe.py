"""
Tic Tac Toe Player
"""

from math import inf
import pytest
from copy import deepcopy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    if sum(row.count(X) for row in board) == sum(row.count(O) for row in board):
        return X
    else:
        return O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    return set(
        [
            (i, j)
            for (i, row) in enumerate(board)
            for (j, item) in enumerate(row)
            if item is EMPTY
        ]
    )


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    if (
        action is None
        or action[0] < 0
        or action[1] < 0
        or action[0] >= len(board)
        or action[1] >= len(board[0])
        or board[action[0]][action[1]] is not EMPTY
    ):
        raise ValueError(f"Invalid action: {action}")

    row, col = action
    new_board = deepcopy(board)
    new_board[row][col] = player(board)

    return new_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    for row in board:
        if row[0] == row[1] == row[2] != EMPTY:
            return row[0]

    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] != EMPTY:
            return board[0][col]

    if board[0][0] == board[1][1] == board[2][2] != EMPTY:
        return board[0][0]
    elif board[0][2] == board[1][1] == board[2][0] != EMPTY:
        return board[0][2]

    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    all_filled = len([col for row in board for col in row if col is EMPTY]) == 0

    return True if all_filled or winner(board) is not None else False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    match winner(board):
        case "X":
            return 1
        case "O":
            return -1
        case None:
            return 0

    print("called utility with no terminal state")
    return None


def min_value(state):
    if terminal(state):
        return utility(state)

    v = inf

    for action in actions(state):
        v = min(v, max_value(result(state, action)))

    return v


def min_action(state):
    v = inf
    a = None

    for action in actions(state):
        prev_v = v
        v = min(v, max_value(result(state, action)))

        if prev_v != v:
            a = action

    return a


def max_value(state):
    if terminal(state):
        return utility(state)

    v = -inf

    for action in actions(state):
        v = max(v, min_value(result(state, action)))

    return v


def max_action(state):
    v = -inf
    a = None

    for action in actions(state):
        prev_v = v
        v = max(v, min_value(result(state, action)))

        if prev_v != v:
            a = action

    return a


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None

    current_player = player(board)

    match current_player:
        case "X":
            if all([row.count(EMPTY) == 3 for row in board]):
                return (1, 1)
            return max_action(board)
        case "O":
            return min_action(board)
