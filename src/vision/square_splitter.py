def split_board(board_image):
    height, width = board_image.shape[:2]

    square_height = height // 8
    square_width = width // 8

    squares = {}

    files = "abcdefgh"

    for row in range(8):
        for col in range(8):
            y1 = row * square_height
            y2 = (row + 1) * square_height

            x1 = col * square_width
            x2 = (col + 1) * square_width

            square = board_image[y1:y2, x1:x2]

            rank = 8 - row
            file = files[col]

            square_name = f"{file}{rank}"

            squares[square_name] = square

    return squares