from piece_detection.board_mapper import detections_to_board


def test_black_rook_is_mapped_to_a8():
    detections = [
        {
            "class_name": "black_rook",
            "confidence": 0.9,
            "bbox": (
                220,
                100,
                280,
                290
            )
        }
    ]

    board = detections_to_board(
        detections,
        board_size=800,
        board_margin=200
    )

    assert board[0][0] == "r"


def test_white_rook_is_mapped_to_h1():
    detections = [
        {
            "class_name": "white_rook",
            "confidence": 0.9,
            "bbox": (
                920,
                800,
                980,
                990
            )
        }
    ]

    board = detections_to_board(
        detections,
        board_size=800,
        board_margin=200
    )

    assert board[7][7] == "R"


def test_higher_confidence_detection_wins():
    detections = [
        {
            "class_name": "white_bishop",
            "confidence": 0.4,
            "bbox": (
                220,
                800,
                280,
                990
            )
        },
        {
            "class_name": "white_rook",
            "confidence": 0.9,
            "bbox": (
                220,
                800,
                280,
                990
            )
        }
    ]

    board = detections_to_board(
        detections,
        board_size=800,
        board_margin=200
    )

    assert board[7][0] == "R"