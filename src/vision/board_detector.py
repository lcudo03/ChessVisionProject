from vision.preprocessing import (
    to_grayscale,
    blur_image,
    detect_edges,
    detect_lines,
    split_lines_by_orientation,
    merge_horizontal_lines,
    merge_vertical_lines,
    horizontal_position,
    vertical_position,
    select_grid_lines,
    find_board_corners,
    warp_board_with_margin
)


def detect_and_warp_board(image, board_size=800, margin=200):
    gray = to_grayscale(image)
    blurred = blur_image(gray)
    edges = detect_edges(blurred)

    lines = detect_lines(edges)

    if lines is None:
        raise ValueError("No lines detected")

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

    horizontal_data = []

    for line in merged_horizontal_lines:
        position = horizontal_position(
            line,
            width / 2
        )

        if position is not None:
            horizontal_data.append(
                (position, line)
            )

    horizontal_data.sort(
        key=lambda item: item[0]
    )

    horizontal_positions = [
        item[0]
        for item in horizontal_data
    ]

    sorted_horizontal_lines = [
        item[1]
        for item in horizontal_data
    ]

    vertical_data = []

    for line in merged_vertical_lines:
        position = vertical_position(
            line,
            height / 2
        )

        if position is not None:
            vertical_data.append(
                (position, line)
            )

    vertical_data.sort(
        key=lambda item: item[0]
    )

    vertical_positions = [
        item[0]
        for item in vertical_data
    ]

    sorted_vertical_lines = [
        item[1]
        for item in vertical_data
    ]

    grid_horizontal_lines = select_grid_lines(
        sorted_horizontal_lines,
        horizontal_positions
    )

    grid_vertical_lines = select_grid_lines(
        sorted_vertical_lines,
        vertical_positions
    )

    if len(grid_horizontal_lines) != 9:
        raise ValueError(
            "Could not detect 9 horizontal grid lines"
        )

    if len(grid_vertical_lines) != 9:
        raise ValueError(
            "Could not detect 9 vertical grid lines"
        )

    corners = find_board_corners(
        grid_horizontal_lines,
        grid_vertical_lines
    )

    if corners is None:
        raise ValueError(
            "Could not determine board corners"
        )

    warped_board = warp_board_with_margin(
        image,
        corners,
        board_size=board_size,
        margin=margin
    )

    return warped_board, corners