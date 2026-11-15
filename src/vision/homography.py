import cv2
import numpy as np


def create_perspective_matrix(
    corners,
    board_size=800,
    margin=200
):
    destination = np.array(
        [
            [margin, margin],
            [margin + board_size, margin],
            [
                margin + board_size,
                margin + board_size
            ],
            [
                margin,
                margin + board_size
            ]
        ],
        dtype=np.float32
    )

    return cv2.getPerspectiveTransform(
        corners,
        destination
    )


def transform_point(
    x,
    y,
    perspective_matrix
):
    point = np.array(
        [[[x, y]]],
        dtype=np.float32
    )

    transformed = cv2.perspectiveTransform(
        point,
        perspective_matrix
    )

    return (
        transformed[0][0][0],
        transformed[0][0][1]
    )