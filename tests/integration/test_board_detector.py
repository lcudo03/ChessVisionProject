from pathlib import Path

import pytest

from vision.preprocessing import load_image
from vision.board_detector import detect_and_warp_board


@pytest.mark.integration
def test_board_can_be_detected_and_warped():
    project_root = Path(__file__).resolve().parent.parent.parent

    image_path = (
        project_root
        / "tests"
        / "assets"
        / "board_start.jpg"
    )

    if not image_path.exists():
        pytest.skip("Test image not available")

    image = load_image(str(image_path))

    warped_board, corners = detect_and_warp_board(
        image,
        board_size=800,
        margin=200
    )

    assert warped_board is not None
    assert warped_board.shape[:2] == (1200, 1200)
    assert len(corners) == 4