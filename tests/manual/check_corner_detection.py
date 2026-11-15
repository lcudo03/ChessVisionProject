from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "chess_corner_detector"
    / "weights"
    / "best.pt"
)

IMAGE_PATH = (
    PROJECT_ROOT
    / "tests"
    / "assets"
    / "chessboard_red2k.jpg"
)

BOARD_SIZE = 800
MARGIN = 200


def create_perspective_matrix(corners):
    destination = np.array(
        [
            [MARGIN, MARGIN],
            [MARGIN + BOARD_SIZE, MARGIN],
            [MARGIN + BOARD_SIZE, MARGIN + BOARD_SIZE],
            [MARGIN, MARGIN + BOARD_SIZE]
        ],
        dtype=np.float32
    )

    return cv2.getPerspectiveTransform(
        corners,
        destination
    )


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
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

    print("Loading corner detector...")

    model = YOLO(
        str(MODEL_PATH)
    )

    print("Detecting corners...")

    results = model.predict(
        source=image,
        conf=0.20,
        iou=0.5,
        device=0,
        verbose=True
    )

    result = results[0]

    print()
    print(
        "All detections:",
        len(result.boxes)
    )

    # Dla każdej klasy zostawiamy tylko
    # detekcję z największym confidence.
    best_detections = {}

    for box in result.boxes:
        class_id = int(
            box.cls[0]
        )

        confidence = float(
            box.conf[0]
        )

        x1, y1, x2, y2 = (
            box.xyxy[0].tolist()
        )

        class_name = (
            model.names[class_id]
        )

        center_x = (
            x1 + x2
        ) / 2

        center_y = (
            y1 + y2
        ) / 2

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
            class_name not in best_detections
            or confidence
            > best_detections[class_name]["confidence"]
        ):
            best_detections[
                class_name
            ] = detection

    print()
    print("Best corner detections:")
    print("=" * 60)

    required_classes = [
        "top_left",
        "top_right",
        "bottom_right",
        "bottom_left"
    ]

    for class_name in required_classes:
        if class_name not in best_detections:
            print(
                f"{class_name:15} NOT DETECTED"
            )

            continue

        detection = (
            best_detections[class_name]
        )

        x, y = detection["center"]

        print(
            f"{class_name:15} "
            f"confidence="
            f"{detection['confidence']:.3f} "
            f"center=({x:.1f}, {y:.1f})"
        )

    annotated = image.copy()

    colors = {
        "top_left": (0, 0, 255),
        "top_right": (0, 255, 0),
        "bottom_right": (255, 0, 0),
        "bottom_left": (0, 255, 255)
    }

    for class_name, detection in best_detections.items():
        if class_name not in colors:
            continue

        x1, y1, x2, y2 = detection["bbox"]

        center_x, center_y = (
            detection["center"]
        )

        color = colors[class_name]

        cv2.rectangle(
            annotated,
            (
                int(x1),
                int(y1)
            ),
            (
                int(x2),
                int(y2)
            ),
            color,
            3
        )

        cv2.circle(
            annotated,
            (
                int(center_x),
                int(center_y)
            ),
            8,
            color,
            -1
        )

        cv2.putText(
            annotated,
            (
                f"{class_name} "
                f"{detection['confidence']:.2f}"
            ),
            (
                int(x1),
                max(
                    int(y1) - 10,
                    20
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    # Jeżeli mamy wszystkie 4 narożniki,
    # testujemy od razu homografię.
    if all(
        class_name in best_detections
        for class_name in required_classes
    ):
        corners = np.array(
            [
                best_detections[
                    "top_left"
                ]["center"],

                best_detections[
                    "top_right"
                ]["center"],

                best_detections[
                    "bottom_right"
                ]["center"],

                best_detections[
                    "bottom_left"
                ]["center"]
            ],
            dtype=np.float32
        )

        perspective_matrix = (
            create_perspective_matrix(
                corners
            )
        )

        output_size = (
            BOARD_SIZE
            + 2 * MARGIN
        )

        warped = cv2.warpPerspective(
            image,
            perspective_matrix,
            (
                output_size,
                output_size
            )
        )

        print()
        print(
            "All four corners detected."
        )

        print(
            "Perspective transform created."
        )

        cv2.imshow(
            "Warped board",
            warped
        )

    else:
        print()
        print(
            "Not all four corners were detected."
        )

    cv2.imshow(
        "Corner detections",
        annotated
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()