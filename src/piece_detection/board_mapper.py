PIECE_TO_SYMBOL = {
    "white_pawn": "P",
    "white_knight": "N",
    "white_bishop": "B",
    "white_rook": "R",
    "white_queen": "Q",
    "white_king": "K",

    "black_pawn": "p",
    "black_knight": "n",
    "black_bishop": "b",
    "black_rook": "r",
    "black_queen": "q",
    "black_king": "k"
}


def detections_to_board(
    detections,
    board_size=800,
    board_margin=200
):
    square_size = board_size / 8

    board = [
        [None for _ in range(8)]
        for _ in range(8)
    ]

    confidences = [
        [0.0 for _ in range(8)]
        for _ in range(8)
    ]

    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]

        bottom_center_x = (x1 + x2) / 2
        bottom_center_y = y2

        relative_x = bottom_center_x - board_margin
        relative_y = bottom_center_y - board_margin

        if not (
            0 <= relative_x < board_size
            and 0 <= relative_y < board_size
        ):
            continue

        col = int(relative_x // square_size)
        row = int(relative_y // square_size)

        confidence = detection["confidence"]

        if confidence > confidences[row][col]:
            symbol = PIECE_TO_SYMBOL[
                detection["class_name"]
            ]

            board[row][col] = symbol
            confidences[row][col] = confidence

    return board