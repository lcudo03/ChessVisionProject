from vision.homography import transform_point

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
    perspective_matrix,
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

        box_height = y2 - y1

        reference_x = (x1 + x2) / 2
        reference_y = y2 - 0.12 * box_height

        transformed_x, transformed_y = transform_point(
            reference_x,
            reference_y,
            perspective_matrix
        )

        relative_x = transformed_x - board_margin
        relative_y = transformed_y - board_margin

        if not (
            0 <= relative_x < board_size
            and 0 <= relative_y < board_size
        ):
            continue

        col = int(relative_x // square_size)
        row = int(relative_y // square_size)

        confidence = detection["confidence"]
        class_name = detection["class_name"]

        if class_name not in PIECE_TO_SYMBOL:
            continue

        if confidence > confidences[row][col]:
            symbol = PIECE_TO_SYMBOL[class_name]

            board[row][col] = symbol
            confidences[row][col] = confidence

    return board