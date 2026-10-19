import cv2

from chess_logic.board import STARTING_BOARD, print_board, validate_board, validate_with_python_chess
from chess_logic.fen import board_to_fen
from chess_logic.evaluator import CustomEvaluator
from chess_engine.stockfish_engine import StockfishAnalyzer

from vision.square_splitter import split_board
from vision.preprocessing import (
    load_image,
    to_grayscale,
    blur_image,
    detect_edges,
    close_edges,
    find_board_contour,
    detect_lines,
    draw_lines,
    split_lines_by_orientation,
    merge_vertical_lines,
    merge_horizontal_lines,
    horizontal_position,
    vertical_position,
    select_grid_lines,
    find_board_corners,
    warp_board
)

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

#Vision tests

image = load_image("../data/board.jpg")

gray = to_grayscale(image)
blurred = blur_image(gray)
edges = detect_edges(blurred)

lines = detect_lines(edges)

horizontal_lines, vertical_lines = split_lines_by_orientation(lines)

height, width = image.shape[:2]

merged_horizontal_lines = merge_horizontal_lines(
    horizontal_lines,
    width
)

merged_vertical_lines = merge_vertical_lines(
    vertical_lines,
    height
)

print("Horizontal lines:", len(horizontal_lines))
print("Merged horizontal:", len(merged_horizontal_lines))

print("Vertical lines:", len(vertical_lines))
print("Merged vertical:", len(merged_vertical_lines))


horizontal_data = []

for line in merged_horizontal_lines:
    position = horizontal_position(
        line,
        width / 2
    )

    if position is not None:
        horizontal_data.append((position, line))

horizontal_data.sort(key=lambda item: item[0])

horizontal_positions = [
    item[0] for item in horizontal_data
]

sorted_horizontal_lines = [
    item[1] for item in horizontal_data
]


vertical_data = []

for line in merged_vertical_lines:
    position = vertical_position(
        line,
        height / 2
    )

    if position is not None:
        vertical_data.append((position, line))

vertical_data.sort(key=lambda item: item[0])

vertical_positions = [
    item[0] for item in vertical_data
]

sorted_vertical_lines = [
    item[1] for item in vertical_data
]


print()
print("Horizontal positions:")
print([round(float(position), 1) for position in horizontal_positions])

print("Horizontal gaps:")
print([
    round(float(horizontal_positions[i + 1] - horizontal_positions[i]), 1)
    for i in range(len(horizontal_positions) - 1)
])


print()
print("Vertical positions:")
print([round(float(position), 1) for position in vertical_positions])

print("Vertical gaps:")
print([
    round(float(vertical_positions[i + 1] - vertical_positions[i]), 1)
    for i in range(len(vertical_positions) - 1)
])


grid_horizontal_lines = select_grid_lines(
    sorted_horizontal_lines,
    horizontal_positions
)

grid_vertical_lines = select_grid_lines(
    sorted_vertical_lines,
    vertical_positions
)

corners = find_board_corners(
    grid_horizontal_lines,
    grid_vertical_lines
)

corner_image = image.copy()

if corners is not None:
    print()
    print("Board corners:")

    for x, y in corners:
        print(round(x, 1), round(y, 1))

        cv2.circle(
            corner_image,
            (int(x), int(y)),
            10,
            (255, 0, 255),
            -1
        )
else:
    print("Could not determine board corners")

print()
print("Selected horizontal grid lines:", len(grid_horizontal_lines))
print("Selected vertical grid lines:", len(grid_vertical_lines))


grid_horizontal_image = draw_lines(
    image,
    grid_horizontal_lines,
    (0, 255, 0)
)

grid_vertical_image = draw_lines(
    image,
    grid_vertical_lines,
    (0, 0, 255)
)


cv2.imshow(
    "Selected horizontal grid",
    grid_horizontal_image
)

cv2.imshow(
    "Selected vertical grid",
    grid_vertical_image
)

if corners is not None:
    for x, y in corners:
        cv2.circle(
            corner_image,
            (int(x), int(y)),
            10,
            (255, 0, 255),
            -1
        )

    warped_board = warp_board(
        image,
        corners,
        size=800
    )

    squares = split_board(warped_board)

    print("Number of squares:", len(squares))
    print("Squares:", list(squares.keys()))

    cv2.imshow("Board corners", corner_image)
    cv2.imshow("Warped board", warped_board)
    cv2.imshow("a8", squares["a8"])
    cv2.imshow("e4", squares["e4"])
    cv2.imshow("h1", squares["h1"])

else:
    print("Could not determine board corners")

cv2.waitKey(0)
cv2.destroyAllWindows()