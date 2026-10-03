import chess
from chess_logic.fen import board_to_fen

# Validation using external function

def validate_with_python_chess(board, active_color="w", castling="-", en_passant="-"):
    fen = board_to_fen(
        board,
        active_color=active_color,
        castling=castling,
        en_passant=en_passant
    )

    try:
        chess_board = chess.Board(fen)
        return chess_board.is_valid()
    except ValueError:
        return False

STARTING_BOARD = [
    ["r", "n", "b", "q", "k", "b", "n", "r"],
    ["p", "p", "p", "p", "p", "p", "p", "p"],
    [None, None, None, None, None, None, None, None],
    [None, None, None, None, None, None, None, None],
    [None, None, None, None, None, None, None, None],
    [None, None, None, None, None, None, None, None],
    ["P", "P", "P", "P", "P", "P", "P", "P"],
    ["R", "N", "B", "Q", "K", "B", "N", "R"]
]

VALID_PIECES = {
    "P", "N", "B", "R", "Q", "K",
    "p", "n", "b", "r", "q", "k"
}


def validate_board(board):
    if len(board) != 8:
        return False

    for row in board:
        if len(row) != 8:
            return False

    white_king_count = 0
    black_king_count = 0

    for row in board:
        for square in row:
            if square is None:
                continue

            if square not in VALID_PIECES:
                return False

            if square == "K":
                white_king_count += 1

            if square == "k":
                black_king_count += 1

    if white_king_count != 1:
        return False

    if black_king_count != 1:
        return False

    return True


def print_board(board):
    for row in board:
        print(" ".join(square if square is not None else "." for square in row))