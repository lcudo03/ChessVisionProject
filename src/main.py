from pathlib import Path

import cv2

from vision.preprocessing import load_image
from vision.board_detector import detect_and_warp_board

from piece_detection.detector import PieceDetector
from piece_detection.board_mapper import detections_to_board

from chess_logic.board import (
    print_board,
    validate_board,
    validate_with_python_chess
)

from chess_logic.fen import board_to_fen
from chess_logic.evaluator import CustomEvaluator


def main():
    project_root = Path(__file__).resolve().parent.parent

    image_path = (
        project_root
        / "tests"
        / "assets"
        / "board_start.jpg"
    )

    model_path = (
        project_root
        / "runs"
        / "chess_pieces_baseline-3"
        / "weights"
        / "best.pt"
    )

    image = load_image(str(image_path))

    warped_board, _ = detect_and_warp_board(
        image,
        board_size=800,
        margin=200
    )

    detector = PieceDetector(
        str(model_path)
    )

    detections = detector.detect(
        warped_board,
        confidence=0.25
    )

    board = detections_to_board(
        detections,
        board_size=800,
        board_margin=200
    )

    print("Recognized board:")
    print_board(board)

    basic_valid = validate_board(board)

    python_chess_valid = validate_with_python_chess(
        board,
        active_color="w",
        castling="-",
        en_passant="-"
    )

    print()
    print("Basic validation:", basic_valid)
    print("Python-chess validation:", python_chess_valid)

    if basic_valid and python_chess_valid:
        fen = board_to_fen(
            board,
            active_color="w",
            castling="-",
            en_passant="-",
            halfmove_clock=0,
            fullmove_number=1
        )

        print()
        print("FEN:", fen)

        evaluator = CustomEvaluator()
        evaluation = evaluator.analyze(board)

        print("Evaluation:", evaluation)

    else:
        print()
        print("Recognized position is invalid.")

    detected_image = detector.draw_detections(
        warped_board,
        confidence=0.25
    )

    cv2.imshow(
        "Detected chess pieces",
        detected_image
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()