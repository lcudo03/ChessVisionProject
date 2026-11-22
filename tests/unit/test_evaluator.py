from chess_logic.board import STARTING_BOARD
from chess_logic.evaluator import CustomEvaluator


def test_starting_board_has_equal_material():
    evaluator = CustomEvaluator()

    result = evaluator.analyze(
        STARTING_BOARD
    )

    assert result["material"] == 0
    assert result["total"] == 0


def test_missing_black_queen_gives_white_advantage():
    board = [
        row.copy()
        for row in STARTING_BOARD
    ]

    board[0][3] = None

    evaluator = CustomEvaluator()

    result = evaluator.analyze(board)

    assert result["material"] == 900
    assert result["total"] == 900


def test_missing_white_rook_gives_black_advantage():
    board = [
        row.copy()
        for row in STARTING_BOARD
    ]

    board[7][0] = None

    evaluator = CustomEvaluator()

    result = evaluator.analyze(board)

    assert result["material"] == -500

from chess_logic.evaluator import CustomEvaluator


def test_position_bonus_for_white_piece_in_center():
    board = [
        [None for _ in range(8)]
        for _ in range(8)
    ]

    board[3][3] = "N"
    board[0][0] = "n"

    evaluator = CustomEvaluator()

    result = evaluator.analyze(board)

    assert result["material"] == 0
    assert result["position"] == 20
    assert result["total"] == 20


def test_position_score_is_equal_for_both_sides():
    board = [
        [None for _ in range(8)]
        for _ in range(8)
    ]

    board[3][3] = "N"
    board[4][4] = "n"

    evaluator = CustomEvaluator()

    result = evaluator.analyze(board)

    assert result["material"] == 0
    assert result["position"] == 0
    assert result["total"] == 0

def test_mobility_from_starting_position():
    evaluator = CustomEvaluator()

    fen = (
        "rnbqkbnr/pppppppp/8/8/"
        "8/8/PPPPPPPP/RNBQKBNR "
        "w KQkq - 0 1"
    )

    result = evaluator.analyze(fen)

    assert result["mobility"] == 0

def test_doubled_white_pawns_are_penalized():
    board = [
        [None for _ in range(8)]
        for _ in range(8)
    ]

    board[5][0] = "P"
    board[6][0] = "P"
    board[6][1] = "P"

    evaluator = CustomEvaluator()

    score = evaluator.evaluate_pawn_structure(
        board
    )

    assert score == -15


def test_isolated_white_pawn_is_penalized():
    board = [
        [None for _ in range(8)]
        for _ in range(8)
    ]

    board[4][3] = "P"

    evaluator = CustomEvaluator()

    score = evaluator.evaluate_pawn_structure(
        board
    )

    assert score == -10


def test_isolated_black_pawn_benefits_white_score():
    board = [
        [None for _ in range(8)]
        for _ in range(8)
    ]

    board[3][3] = "p"

    evaluator = CustomEvaluator()

    score = evaluator.evaluate_pawn_structure(
        board
    )

    assert score == 10

def test_white_king_pawn_shield():
    board = [
        [None for _ in range(8)]
        for _ in range(8)
    ]

    board[7][6] = "K"

    board[6][5] = "P"
    board[6][6] = "P"
    board[6][7] = "P"

    evaluator = CustomEvaluator()

    score = evaluator.evaluate_king_safety(
        board
    )

    assert score == 30

def test_white_king_in_center_is_penalized():
    board = [
        [None for _ in range(8)]
        for _ in range(8)
    ]

    board[4][4] = "K"

    evaluator = CustomEvaluator()

    score = evaluator.evaluate_king_safety(
        board
    )

    assert score == -20

def test_black_king_pawn_shield():
    board = [
        [None for _ in range(8)]
        for _ in range(8)
    ]

    board[0][6] = "k"

    board[1][5] = "p"
    board[1][6] = "p"
    board[1][7] = "p"

    evaluator = CustomEvaluator()

    score = evaluator.evaluate_king_safety(
        board
    )

    assert score == -30

def test_best_move_is_returned():
    evaluator = CustomEvaluator()

    fen = (
        "rnbqkbnr/pppppppp/8/8/"
        "8/8/PPPPPPPP/RNBQKBNR "
        "w KQkq - 0 1"
    )

    result = evaluator.analyze(fen)

    assert result["best_move"] is not None

import chess


def test_negamax_finds_checkmate_in_one():
    evaluator = CustomEvaluator()

    fen = "7k/8/5KQ1/8/8/8/8/8 w - - 0 1"

    result = evaluator.analyze(fen)

    assert result["best_move"] is not None

    board = chess.Board(fen)

    move = chess.Move.from_uci(
        result["best_move"]
    )

    assert move in board.legal_moves

    board.push(move)

    assert board.is_checkmate()