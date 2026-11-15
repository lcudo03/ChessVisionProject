from pathlib import Path

import cv2
import pytest

from pipeline import ChessVisionPipeline


@pytest.mark.integration
def test_complete_pipeline_generates_expected_fen():
    project_root = Path(__file__).resolve().parent.parent.parent

    image_path = (
        project_root
        / "tests"
        / "assets"
        / "chessboard_red2k.jpg"
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

    if not image_path.exists():
        pytest.skip("Test image not available")

    if not piece_model_path.exists():
        pytest.skip("Piece detector model not available")

    if not corner_model_path.exists():
        pytest.skip("Corner detector model not available")

    image = cv2.imread(str(image_path))

    assert image is not None

    pipeline = ChessVisionPipeline(
        piece_model_path=piece_model_path,
        corner_model_path=corner_model_path
    )

    result = pipeline.recognize(image)

    expected_fen = (
        "2r3rk/2pn3p/1p3b1q/"
        "pN1PpN1n/2P2p2/1P4P1/"
        "PB2Q2P/4RR1K w - - 0 1"
    )

    assert result["fen"] == expected_fen