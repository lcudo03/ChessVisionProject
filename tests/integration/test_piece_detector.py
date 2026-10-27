from pathlib import Path

import pytest

from vision.preprocessing import load_image
from vision.board_detector import detect_and_warp_board
from piece_detection.detector import PieceDetector


@pytest.mark.integration
def test_piece_detector_returns_detections():
    project_root = Path(__file__).resolve().parent.parent.parent

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

    if not image_path.exists():
        pytest.skip("Test image not available")

    if not model_path.exists():
        pytest.skip("YOLO model not available")

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

    assert len(detections) > 0

    for detection in detections:
        assert "class_name" in detection
        assert "confidence" in detection
        assert "bbox" in detection

        assert 0 <= detection["confidence"] <= 1
        assert len(detection["bbox"]) == 4