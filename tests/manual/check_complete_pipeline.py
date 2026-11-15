from pathlib import Path
import sys

import cv2
import numpy as np
from ultralytics import YOLO


project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"

sys.path.insert(0, str(src_path))

from piece_detection.board_mapper import detections_to_board


BOARD_SIZE = 800
BOARD_MARGIN = 200


CORNER_MODEL_PATH = (
    project_root
    / "runs"
    / "chess_corner_detector"
    / "weights"
    / "best.pt"
)

PIECE_MODEL_PATH = (
    project_root
    / "runs"
    / "chess_pieces_v2_withChessRedDataset-3"
    / "weights"
    / "best.pt"
)

IMAGE_PATH = (
    project_root
    / "tests"
    / "assets"
    / "chessboard_red2k.jpg"
)


def get_best_corner_detections(result, model):
    best = {}

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        class_name = model.names[class_id]

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        detection = {
            "confidence": confidence,
            "center": (
                center_x,
                center_y
            ),
            "bbox": (
                x1,
                y1,
                x2,
                y2
            )
        }

        if (
            class_name not in best
            or confidence > best[class_name]["confidence"]
        ):
            best[class_name] = detection

    return best


def create_perspective_matrix(corners):
    destination = np.array(
        [
            [
                BOARD_MARGIN,
                BOARD_MARGIN
            ],
            [
                BOARD_MARGIN + BOARD_SIZE,
                BOARD_MARGIN
            ],
            [
                BOARD_MARGIN + BOARD_SIZE,
                BOARD_MARGIN + BOARD_SIZE
            ],
            [
                BOARD_MARGIN,
                BOARD_MARGIN + BOARD_SIZE
            ]
        ],
        dtype=np.float32
    )

    return cv2.getPerspectiveTransform(
        corners,
        destination
    )


def create_piece_detections(result, model):
    detections = []

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        class_name = model.names[class_id]

        detections.append(
            {
                "class_name": class_name,
                "confidence": confidence,
                "bbox": [
                    x1,
                    y1,
                    x2,
                    y2
                ]
            }
        )

    return detections


def board_to_fen(board):
    rows = []

    for row in board:
        empty = 0
        fen_row = ""

        for square in row:
            if square is None:
                empty += 1
            else:
                if empty > 0:
                    fen_row += str(empty)
                    empty = 0

                fen_row += square

        if empty > 0:
            fen_row += str(empty)

        rows.append(fen_row)

    return "/".join(rows)


def print_board(board):
    print()
    print("Detected board:")
    print()

    for row in board:
        print(
            " ".join(
                square if square is not None else "."
                for square in row
            )
        )


def draw_corners(image, corners):
    output = image.copy()

    colors = {
        "top_left": (0, 0, 255),
        "top_right": (0, 255, 0),
        "bottom_right": (255, 0, 0),
        "bottom_left": (0, 255, 255)
    }

    for class_name, detection in corners.items():
        if class_name not in colors:
            continue

        x, y = detection["center"]
        color = colors[class_name]

        cv2.circle(
            output,
            (
                int(x),
                int(y)
            ),
            10,
            color,
            -1
        )

        cv2.putText(
            output,
            (
                f"{class_name} "
                f"{detection['confidence']:.2f}"
            ),
            (
                int(x) + 10,
                int(y) - 10
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    return output


def main():
    if not CORNER_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Corner model not found: {CORNER_MODEL_PATH}"
        )

    if not PIECE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Piece model not found: {PIECE_MODEL_PATH}"
        )

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Image not found: {IMAGE_PATH}"
        )

    image = cv2.imread(
        str(IMAGE_PATH)
    )

    if image is None:
        raise RuntimeError(
            "Could not load image"
        )

    # =========================================================
    # 1. CORNER DETECTION
    # =========================================================

    print("1. Loading corner detector...")

    corner_model = YOLO(
        str(CORNER_MODEL_PATH)
    )

    print("2. Detecting board corners...")

    corner_results = corner_model.predict(
        source=image,
        conf=0.20,
        iou=0.5,
        device=0,
        verbose=False
    )

    corner_result = corner_results[0]

    best_corners = get_best_corner_detections(
        corner_result,
        corner_model
    )

    required_corners = [
        "top_left",
        "top_right",
        "bottom_right",
        "bottom_left"
    ]

    print()
    print("Corner detections:")
    print("=" * 60)

    for name in required_corners:
        if name not in best_corners:
            raise RuntimeError(
                f"Corner not detected: {name}"
            )

        detection = best_corners[name]

        print(
            f"{name:15} "
            f"confidence={detection['confidence']:.3f} "
            f"center={detection['center']}"
        )

    # Klasy corner datasetu odpowiadają:
    #
    # top_left     -> a8
    # top_right    -> h8
    # bottom_right -> h1
    # bottom_left  -> a1
    #
    # Dzięki temu po homografii otrzymujemy
    # standardową orientację szachową.

    corners = np.array(
        [
            best_corners[
                "top_left"
            ]["center"],

            best_corners[
                "top_right"
            ]["center"],

            best_corners[
                "bottom_right"
            ]["center"],

            best_corners[
                "bottom_left"
            ]["center"]
        ],
        dtype=np.float32
    )

    perspective_matrix = create_perspective_matrix(
        corners
    )

    output_size = (
        BOARD_SIZE
        + 2 * BOARD_MARGIN
    )

    warped_board = cv2.warpPerspective(
        image,
        perspective_matrix,
        (
            output_size,
            output_size
        )
    )

    # =========================================================
    # 2. PIECE DETECTION
    # =========================================================

    print()
    print("3. Loading piece detector...")

    piece_model = YOLO(
        str(PIECE_MODEL_PATH)
    )

    print("4. Detecting chess pieces...")

    piece_results = piece_model.predict(
        source=image,
        conf=0.25,
        iou=0.5,
        agnostic_nms=True,
        device=0,
        verbose=False
    )

    piece_result = piece_results[0]

    print()
    print(
        "Detected pieces:",
        len(piece_result.boxes)
    )

    detections = create_piece_detections(
        piece_result,
        piece_model
    )

    for detection in detections:
        print(
            f"{detection['class_name']:15} "
            f"confidence="
            f"{detection['confidence']:.3f}"
        )

    # =========================================================
    # 3. BOARD MAPPING
    # =========================================================

    print()
    print("5. Mapping pieces to chessboard...")

    board = detections_to_board(
        detections,
        perspective_matrix,
        board_size=BOARD_SIZE,
        board_margin=BOARD_MARGIN
    )

    print_board(
        board
    )

    # =========================================================
    # 4. FEN
    # =========================================================

    fen = board_to_fen(
        board
    )

    print()
    print("Generated FEN:")
    print(fen)

    # =========================================================
    # 5. VISUALIZATION
    # =========================================================

    corner_preview = draw_corners(
        image,
        best_corners
    )

    piece_preview = piece_result.plot()

    cv2.imshow(
        "1 - Detected corners",
        corner_preview
    )

    cv2.imshow(
        "2 - Warped board",
        warped_board
    )

    cv2.imshow(
        "3 - Detected pieces",
        piece_preview
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()