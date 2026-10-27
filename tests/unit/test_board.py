from chess_logic.board import STARTING_BOARD, validate_board


def test_starting_board_is_valid():
    assert validate_board(STARTING_BOARD)


def test_board_without_black_king_is_invalid():
    board = [
        row.copy()
        for row in STARTING_BOARD
    ]

    board[0][4] = None

    assert not validate_board(board)


def test_board_without_white_king_is_invalid():
    board = [
        row.copy()
        for row in STARTING_BOARD
    ]

    board[7][4] = None

    assert not validate_board(board)


def test_board_with_invalid_number_of_rows_is_invalid():
    board = [
        [None] * 8
        for _ in range(7)
    ]

    assert not validate_board(board)


def test_board_with_invalid_number_of_columns_is_invalid():
    board = [
        [None] * 8
        for _ in range(8)
    ]

    board[0] = [None] * 7

    assert not validate_board(board)