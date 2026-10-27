from chess_logic.board import STARTING_BOARD
from chess_logic.fen import (
    board_to_fen,
    board_to_fen_position
)


def test_starting_board_position_fen():
    fen = board_to_fen_position(
        STARTING_BOARD
    )

    assert (
        fen
        == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    )


def test_starting_board_full_fen():
    fen = board_to_fen(
        STARTING_BOARD,
        active_color="w",
        castling="KQkq",
        en_passant="-",
        halfmove_clock=0,
        fullmove_number=1
    )

    assert (
        fen
        == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR "
           "w KQkq - 0 1"
    )