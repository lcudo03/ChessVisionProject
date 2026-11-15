from piece_detection.detector import PieceDetector
from piece_detection.board_mapper import detections_to_board
from vision.corner_detector import CornerDetector
from vision.homography import create_perspective_matrix
from chess_logic.fen import board_to_fen


class ChessVisionPipeline:
    def __init__(
        self,
        piece_model_path,
        corner_model_path,
        board_size=800,
        board_margin=200
    ):
        self.board_size = board_size
        self.board_margin = board_margin

        self.piece_detector = PieceDetector(
            piece_model_path
        )

        self.corner_detector = CornerDetector(
            corner_model_path
        )

    def recognize(self, image):
        corner_detections = self.corner_detector.detect(
            image
        )

        corner_points = self.corner_detector.get_corner_points(
            corner_detections
        )

        perspective_matrix = create_perspective_matrix(
            corner_points,
            board_size=self.board_size,
            margin=self.board_margin
        )

        piece_detections = self.piece_detector.detect(
            image,
            confidence=0.25
        )

        board = detections_to_board(
            piece_detections,
            perspective_matrix,
            board_size=self.board_size,
            board_margin=self.board_margin
        )

        fen = board_to_fen(
            board
        )

        return {
            "board": board,
            "fen": fen,
            "corners": corner_detections,
            "pieces": piece_detections,
            "perspective_matrix": perspective_matrix
        }

def print_board(board):
    for row in board:
        print(
            " ".join(
                square if square is not None else "."
                for square in row
            )
        )