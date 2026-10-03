def board_to_fen_position(board):
    fen_rows = []

    for row in board:
        fen_row = ""
        empty_count = 0

        for square in row:
            if square is None:
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0

                fen_row += square

        if empty_count > 0:
            fen_row += str(empty_count)

        fen_rows.append(fen_row)

    return "/".join(fen_rows)

def board_to_fen(
    board,
    active_color="w",
    castling="-",
    en_passant="-",
    halfmove_clock=0,
    fullmove_number=1
):
    position = board_to_fen_position(board)

    return (
        f"{position} "
        f"{active_color} "
        f"{castling} "
        f"{en_passant} "
        f"{halfmove_clock} "
        f"{fullmove_number}"
    )