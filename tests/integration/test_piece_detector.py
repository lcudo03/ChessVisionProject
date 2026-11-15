from pathlib import Path

import cv2
import pytest

from piece_detection.detector import PieceDetector


@pytest.mark.integration
def test_piece_detector_returns_detections():
    project_root = Path(__file__).resolve().parent.parent.parent

    image_path = (
        project_root
        / "tests"
        / "assets"
        / "chessboard_red2k.jpg"
    )

    model_path = (
        project_root
        / "models"
        / "piece_detector"
        / "best.pt"
    )

    if not image_path.exists():
        pytest.skip("Test image not available")

    if not model_path.exists():
        pytest.skip("YOLO model not available")

    image = cv2.imread(str(image_path))

    assert image is not None

    detector = PieceDetector(
        str(model_path)
    )

    detections = detector.detect(
        image,
        confidence=0.25
    )

    assert isinstance(detections, list)
    assert len(detections) > 0

    for detection in detections:
        assert "class_name" in detection
        assert "confidence" in detection
        assert "bbox" in detection