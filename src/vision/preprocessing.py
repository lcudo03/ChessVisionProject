import cv2
import numpy as np
from itertools import combinations

def load_image(path):
    image = cv2.imread(path)

    if image is None:
        raise ValueError(f"Could not load image: {path}")

    return image


def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def blur_image(image):
    return cv2.GaussianBlur(image, (5, 5), 0)


def detect_edges(image):
    return cv2.Canny(image, 50, 150)

def find_board_contour(edges):
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    height, width = edges.shape
    image_area = width * height

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < image_area * 0.15:
            continue

        perimeter = cv2.arcLength(contour, True)

        approx = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True
        )

        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx

    return None

def close_edges(edges):
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (5, 5)
    )

    return cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

def detect_lines(edges):
    height, width = edges.shape

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=int(width * 0.12),
        maxLineGap=30
    )

    return lines

def draw_lines(image, lines, color=(0, 255, 0)):
    result = image.copy()

    if lines is None:
        return result

    for line in lines:
        x1, y1, x2, y2 = np.asarray(line).reshape(-1)[:4]

        cv2.line(
            result,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            color,
            2
        )

    return result

def split_lines_by_orientation(lines):
    horizontal_lines = []
    vertical_lines = []

    if lines is None:
        return horizontal_lines, vertical_lines

    for line in lines:
        x1, y1, x2, y2 = np.asarray(line).reshape(-1)[:4]

        angle = np.degrees(
            np.arctan2(y2 - y1, x2 - x1)
        )

        angle = abs(angle)

        if angle > 90:
            angle = 180 - angle

        if angle < 20:
            horizontal_lines.append((x1, y1, x2, y2))

        elif angle > 60:
            vertical_lines.append((x1, y1, x2, y2))

    return horizontal_lines, vertical_lines

def line_length(line):
    x1, y1, x2, y2 = line

    return np.sqrt(
        (x2 - x1) ** 2 +
        (y2 - y1) ** 2
    )


def horizontal_position(line, x_reference):
    x1, y1, x2, y2 = line

    if x2 == x1:
        return None

    return y1 + (
        (x_reference - x1) *
        (y2 - y1) /
        (x2 - x1)
    )


def vertical_position(line, y_reference):
    x1, y1, x2, y2 = line

    if y2 == y1:
        return None

    return x1 + (
        (y_reference - y1) *
        (x2 - x1) /
        (y2 - y1)
    )


def merge_horizontal_lines(lines, image_width, tolerance=12):
    reference_x = image_width / 2

    positioned_lines = []

    for line in lines:
        position = horizontal_position(line, reference_x)

        if position is not None:
            positioned_lines.append((position, line))

    positioned_lines.sort(key=lambda item: item[0])

    merged = []

    for position, line in positioned_lines:
        if not merged:
            merged.append((position, line))
            continue

        previous_position, previous_line = merged[-1]

        if abs(position - previous_position) < tolerance:
            if line_length(line) > line_length(previous_line):
                merged[-1] = (position, line)
        else:
            merged.append((position, line))

    return [line for _, line in merged]


def merge_vertical_lines(lines, image_height, tolerance=12):
    reference_y = image_height / 2

    positioned_lines = []

    for line in lines:
        position = vertical_position(line, reference_y)

        if position is not None:
            positioned_lines.append((position, line))

    positioned_lines.sort(key=lambda item: item[0])

    merged = []

    for position, line in positioned_lines:
        if not merged:
            merged.append((position, line))
            continue

        previous_position, previous_line = merged[-1]

        if abs(position - previous_position) < tolerance:
            if line_length(line) > line_length(previous_line):
                merged[-1] = (position, line)
        else:
            merged.append((position, line))

    return [line for _, line in merged]

def select_grid_lines(lines, positions, count=9):
    if len(lines) < count:
        return []

    best_score = None
    best_indices = None

    for indices in combinations(range(len(lines)), count):
        selected_positions = np.array(
            [positions[i] for i in indices],
            dtype=float
        )

        gaps = np.diff(selected_positions)

        if np.any(gaps <= 0):
            continue

        mean_gap = np.mean(gaps)

        if mean_gap == 0:
            continue

        gap_variation = np.std(gaps) / mean_gap

        if len(gaps) > 1:
            smoothness = (
                np.mean(np.abs(np.diff(gaps)))
                / np.median(gaps)
            )
        else:
            smoothness = 0

        score = smoothness + 0.3 * gap_variation

        if best_score is None or score < best_score:
            best_score = score
            best_indices = indices

    if best_indices is None:
        return []

    return [lines[i] for i in best_indices]

def line_intersection(line1, line2):
    x1, y1, x2, y2 = map(float, line1)
    x3, y3, x4, y4 = map(float, line2)

    denominator = (
        (x1 - x2) * (y3 - y4)
        - (y1 - y2) * (x3 - x4)
    )

    if abs(denominator) < 1e-10:
        return None

    determinant1 = x1 * y2 - y1 * x2
    determinant2 = x3 * y4 - y3 * x4

    x = (
        determinant1 * (x3 - x4)
        - (x1 - x2) * determinant2
    ) / denominator

    y = (
        determinant1 * (y3 - y4)
        - (y1 - y2) * determinant2
    ) / denominator

    return x, y

def find_board_corners(horizontal_lines, vertical_lines):
    if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
        return None

    top = horizontal_lines[0]
    bottom = horizontal_lines[-1]

    left = vertical_lines[0]
    right = vertical_lines[-1]

    top_left = line_intersection(top, left)
    top_right = line_intersection(top, right)
    bottom_right = line_intersection(bottom, right)
    bottom_left = line_intersection(bottom, left)

    if (
        top_left is None
        or top_right is None
        or bottom_right is None
        or bottom_left is None
    ):
        return None

    return [
        top_left,
        top_right,
        bottom_right,
        bottom_left
    ]

def warp_board(image, corners, size=800):
    source_points = np.float32(corners)

    destination_points = np.float32([
        [0, 0],
        [size - 1, 0],
        [size - 1, size - 1],
        [0, size - 1]
    ])

    transformation_matrix = cv2.getPerspectiveTransform(
        source_points,
        destination_points
    )

    warped = cv2.warpPerspective(
        image,
        transformation_matrix,
        (size, size)
    )

    return warped