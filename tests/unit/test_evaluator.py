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