from chess_logic.board import STARTING_BOARD, print_board, validate_board, validate_with_python_chess
from chess_logic.fen import board_to_fen
from chess_logic.evaluator import CustomEvaluator
from chess_engine.stockfish_engine import StockfishAnalyzer

# Board print
print_board(STARTING_BOARD)
print()
# Board validation minitests
print("Board valid:", validate_board(STARTING_BOARD))

fen = board_to_fen(
    STARTING_BOARD,
    active_color="w",
    castling="KQkq"
)

invalid_board = [
    ["r", "n", "b", "q", None, "b", "n", "r"],
    ["p", "p", "p", "p", "p", "p", "p", "p"],
    [None, None, None, None, None, None, None, None],
    [None, None, None, None, None, None, None, None],
    [None, None, None, None, None, None, None, None],
    [None, None, None, None, None, None, None, None],
    ["P", "P", "P", "P", "P", "P", "P", "P"],
    ["R", "N", "B", "Q", "K", "B", "N", "R"]
]

print("Board valid(false):", validate_board(invalid_board))

# FEN notation position
print("FEN:", fen)

#Usage of custom evaluator
evaluator = CustomEvaluator()
result = evaluator.analyze(STARTING_BOARD)
print(result)


# Evaluating test (no black Queen)
test_board = [
    ["r", "n", "b", None, "k", "b", "n", "r"],
    ["p", "p", "p", "p", "p", "p", "p", "p"],
    [None, None, None, None, None, None, None, None],
    [None, None, None, None, None, None, None, None],
    [None, None, None, None, None, None, None, None],
    [None, None, None, None, None, None, None, None],
    ["P", "P", "P", "P", "P", "P", "P", "P"],
    ["R", "N", "B", "Q", "K", "B", "N", "R"]
]

print(evaluator.analyze(test_board))

#Basic validation using external and internal function
print("Basic validation:", validate_board(STARTING_BOARD))

print("Python-chess validation:", validate_with_python_chess( STARTING_BOARD, active_color="w", castling="KQkq"))


#Invalid one
invalid_board = [
    ["r", "n", "b", "q", None, "b", "n", "r"],
    ["p", "p", "p", "p", "p", "p", "p", "p"],
    [None, None, None, None, None, None, None, None],
    [None, None, None, None, None, None, None, None],
    [None, None, None, None, None, None, None, None],
    [None, None, None, None, None, None, None, None],
    ["P", "P", "P", "P", "P", "P", "P", "P"],
    ["R", "N", "B", "Q", "K", "B", "N", "R"]
]

print("Basic:", validate_board(invalid_board))
print("Python-chess:", validate_with_python_chess(invalid_board))

# StockFish tests
custom = CustomEvaluator()

custom_result = custom.analyze(STARTING_BOARD)

print("Custom evaluator:")
print(custom_result)


stockfish = StockfishAnalyzer(
    "../engine/stockfish.exe"
)

stockfish_result = stockfish.analyze(STARTING_BOARD)

print()
print("Stockfish:")
print(stockfish_result)

stockfish.close()