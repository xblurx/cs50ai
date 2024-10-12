import pytest
from copy import deepcopy
from tictactoe import (
    player,
    actions,
    result,
    winner,
    terminal,
    utility,
)


EMPTY = None

X = "X"
O = "O"


def test_actions_empty_board():
    board = [[EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY]]
    assert actions(board) == {
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
        (2, 0),
        (2, 1),
        (2, 2),
    }


def test_actions_partially_filled_board():
    board = [[EMPTY, "X", EMPTY], [EMPTY, EMPTY, EMPTY], ["O", EMPTY, EMPTY]]
    assert actions(board) == {(0, 0), (0, 2), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2)}


def test_actions_full_board():
    board = [["X", "O", "X"], ["O", "X", "O"], ["X", "O", "X"]]
    assert actions(board) == set()


def test_result_valid_action():
    board = [[EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY]]
    new_board = result(board, (0, 0))
    assert new_board == [
        [X, EMPTY, EMPTY],
        [EMPTY, EMPTY, EMPTY],
        [EMPTY, EMPTY, EMPTY],
    ]


def test_result_invalid_action():
    board = [[X, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY]]
    with pytest.raises(ValueError):
        result(board, (0, 0))


def test_result_does_not_modify_original_board():
    board = [[EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY]]
    original_board = deepcopy(board)
    new_board = result(board, (0, 0))
    assert board == original_board
    assert new_board != board


def test_result_alternates_turns():
    board = [[EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY]]
    new_board1 = result(board, (0, 0))
    new_board2 = result(new_board1, (1, 1))
    assert new_board1 == [
        [X, EMPTY, EMPTY],
        [EMPTY, EMPTY, EMPTY],
        [EMPTY, EMPTY, EMPTY],
    ]
    assert new_board2 == [[X, EMPTY, EMPTY], [EMPTY, O, EMPTY], [EMPTY, EMPTY, EMPTY]]


def test_result_throws_out_of_bounds_action():
    board = [[EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY]]
    with pytest.raises(ValueError):
        new_board = result(board, (9, 12))


def test_winner_x_wins_row():
    board = [[X, X, X], [O, O, EMPTY], [EMPTY, EMPTY, EMPTY]]
    assert winner(board) == X


def test_winner_x_wins_column():
    board = [[O, X, X], [O, X, EMPTY], [EMPTY, X, O]]
    assert winner(board) == X


def test_winner_x_wins_diagonal():
    board = [[X, O, O], [O, X, EMPTY], [EMPTY, EMPTY, X]]
    assert winner(board) == X


def test_winner_no_winner():
    board = [[X, O, X], [O, X, O], [O, X, EMPTY]]
    assert winner(board) is None


def test_terminal_game_over_x_wins():
    board = [[X, X, X], [O, O, EMPTY], [EMPTY, EMPTY, EMPTY]]
    assert terminal(board)


def test_terminal_game_over_o_wins():
    board = [[O, O, O], [X, X, EMPTY], [EMPTY, EMPTY, EMPTY]]
    assert terminal(board)


def test_terminal_game_over_tie():
    board = [[X, O, X], [O, X, O], [O, X, O]]
    assert terminal(board)


def test_terminal_game_in_progress():
    board = [[X, O, EMPTY], [EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY]]
    assert not terminal(board)


def test_terminal_game_idk():
    board = [[EMPTY, O, X], [O, O, X], [O, X, X]]
    assert terminal(board)


def test_utility_x_wins():
    board = [[X, X, X], [O, O, EMPTY], [EMPTY, EMPTY, EMPTY]]
    assert utility(board) == 1


def test_utility_o_wins():
    board = [[O, O, O], [X, X, EMPTY], [EMPTY, EMPTY, EMPTY]]
    assert utility(board) == -1


def test_utility_tie():
    board = [[X, O, X], [O, X, O], [O, X, O]]
    assert utility(board) == 0
