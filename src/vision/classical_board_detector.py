import cv2
import numpy as np

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


def order_corners(points):
    points = np.array(points, dtype=np.float32)

    ordered = np.zeros((4, 2), dtype=np.float32)

    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).flatten()

    ordered[0] = points[np.argmin(sums)]          # top-left
    ordered[2] = points[np.argmax(sums)]          # bottom-right

    ordered[1] = points[np.argmin(differences)]   # top-right
    ordered[3] = points[np.argmax(differences)]   # bottom-left

    return ordered


def create_perspective_matrix(
    corners,
    board_size=800,
    margin=200
):
    source_points = order_corners(corners)

    destination_points = np.array(
        [
            [margin, margin],
            [margin + board_size, margin],
            [margin + board_size, margin + board_size],
            [margin, margin + board_size]
        ],
        dtype=np.float32
    )

    return cv2.getPerspectiveTransform(
        source_points,
        destination_points
    )


def detect_board_from_grid(image):
    gray = to_grayscale(image)
    blurred = blur_image(gray)
    edges = detect_edges(blurred)

    lines = detect_lines(edges)

    if lines is None:
        return None

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
        return None

    if len(grid_vertical_lines) != 9:
        return None

    corners = find_board_corners(
        grid_horizontal_lines,
        grid_vertical_lines
    )

    if corners is None:
        return None

    return order_corners(corners)


def detect_board_from_contours(image):
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        blurred,
        50,
        150
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (5, 5)
    )

    closed = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    contours, _ = cv2.findContours(
        closed,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    height, width = image.shape[:2]
    image_area = width * height

    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < image_area * 0.10:
            continue

        perimeter = cv2.arcLength(
            contour,
            True
        )

        approximation = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True
        )

        if len(approximation) != 4:
            continue

        if not cv2.isContourConvex(approximation):
            continue

        points = approximation.reshape(4, 2)

        ordered = order_corners(points)

        top_left = ordered[0]
        top_right = ordered[1]
        bottom_right = ordered[2]
        bottom_left = ordered[3]

        top_width = np.linalg.norm(
            top_right - top_left
        )

        bottom_width = np.linalg.norm(
            bottom_right - bottom_left
        )

        left_height = np.linalg.norm(
            bottom_left - top_left
        )

        right_height = np.linalg.norm(
            bottom_right - top_right
        )

        average_width = (
            top_width + bottom_width
        ) / 2

        average_height = (
            left_height + right_height
        ) / 2

        if average_width == 0 or average_height == 0:
            continue

        aspect_ratio = (
            average_width / average_height
        )

        if not 0.5 < aspect_ratio < 2.0:
            continue

        candidates.append(
            (
                area,
                ordered
            )
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return candidates[0][1]


def warp_with_matrix(
    image,
    perspective_matrix,
    board_size=800,
    margin=200
):
    output_size = (
        board_size + 2 * margin
    )

    return cv2.warpPerspective(
        image,
        perspective_matrix,
        (
            output_size,
            output_size
        )
    )


def detect_and_warp_board(
    image,
    board_size=800,
    margin=200
):
    print(
        "Trying grid-based board detection..."
    )

    corners = detect_board_from_grid(
        image
    )

    if corners is not None:
        print(
            "Board detected using grid lines."
        )

    else:
        print(
            "Grid detection failed."
        )

        print(
            "Trying contour-based board detection..."
        )

        corners = detect_board_from_contours(
            image
        )

        if corners is None:
            raise ValueError(
                "Could not detect chessboard "
                "using grid lines or contours"
            )

        print(
            "Board detected using outer contour."
        )

    perspective_matrix = create_perspective_matrix(
        corners,
        board_size=board_size,
        margin=margin
    )

    warped_board = warp_with_matrix(
        image,
        perspective_matrix,
        board_size=board_size,
        margin=margin
    )

    return (
        warped_board,
        corners,
        perspective_matrix
    )