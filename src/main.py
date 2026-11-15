from pathlib import Path

import cv2

from pipeline import (
    ChessVisionPipeline,
    print_board
)


def main():
    project_root = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    piece_model_path = (
        project_root
        / "models"
        / "piece_detector"
        / "best.pt"
    )

    corner_model_path = (
        project_root
        / "models"
        / "corner_detector"
        / "best.pt"
    )

    image_path = (
        project_root
        / "tests"
        / "assets"
        / "chessboard_red2k.jpg"
    )

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {image_path}"
        )

    pipeline = ChessVisionPipeline(
        piece_model_path=piece_model_path,
        corner_model_path=corner_model_path
    )

    result = pipeline.recognize(
        image
    )

    print()
    print("Detected board:")
    print()

    print_board(
        result["board"]
    )

    print()
    print("FEN:")
    print(
        result["fen"]
    )


if __name__ == "__main__":
    main()