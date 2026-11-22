from pathlib import Path

import chess
import cv2

from pipeline import ChessVisionPipeline
from chess_logic.evaluator import CustomEvaluator
from chess_engine.stockfish_engine import StockfishAnalyzer


project_root = Path(__file__).resolve().parent.parent

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

stockfish_path = (
    project_root
    / "engine"
    / "stockfish.exe"
)


def print_board(board):
    for row in board:
        print(
            " ".join(
                square
                if square is not None
                else "."
                for square in row
            )
        )


def uci_line_to_san(
    fen,
    line
):
    board = chess.Board(fen)

    san_line = []

    for uci_move in line:
        move = chess.Move.from_uci(
            uci_move
        )

        if move not in board.legal_moves:
            break

        san_line.append(
            board.san(move)
        )

        board.push(move)

    return san_line


def main():
    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise ValueError(
            f"Could not load image: "
            f"{image_path}"
        )

    pipeline = ChessVisionPipeline(
        piece_model_path=piece_model_path,
        corner_model_path=corner_model_path
    )

    result = pipeline.recognize(
        image,
        active_color="w"
    )

    fen = result["fen"]

    print("=" * 60)
    print("DETECTED POSITION")
    print("=" * 60)

    print_board(
        result["board"]
    )

    print()
    print("FEN:")
    print(fen)

    custom_evaluator = CustomEvaluator()

    custom_analysis = (
        custom_evaluator.analyze(
            fen
        )
    )

    print()
    print("=" * 60)
    print("CUSTOM ENGINE ANALYSIS")
    print("=" * 60)

    print(
        f"Material: "
        f"{custom_analysis['material']}"
    )

    print(
        f"Position: "
        f"{custom_analysis['position']}"
    )

    print(
        f"Mobility: "
        f"{custom_analysis['mobility']}"
    )

    print(
        f"Pawn structure: "
        f"{custom_analysis['pawn_structure']}"
    )

    print(
        f"King safety: "
        f"{custom_analysis['king_safety']}"
    )

    print(
        f"Strategic: "
        f"{custom_analysis['strategic']}"
    )

    print()

    print(
        f"Static evaluation: "
        f"{custom_analysis['total'] / 100:+.2f}"
    )

    print(
        f"Best move (UCI): "
        f"{custom_analysis['best_move']}"
    )

    if (
        custom_analysis[
            "search_evaluation"
        ]
        is not None
    ):
        print(
            f"Search evaluation: "
            f"{custom_analysis['search_evaluation'] / 100:+.2f}"
        )

    if custom_analysis["best_line"]:
        custom_san_line = (
            uci_line_to_san(
                fen,
                custom_analysis[
                    "best_line"
                ]
            )
        )

        print(
            f"Best line (UCI): "
            f"{' '.join(custom_analysis['best_line'])}"
        )

        print(
            f"Best line (SAN): "
            f"{' '.join(custom_san_line)}"
        )

    stockfish = StockfishAnalyzer(
        engine_path=stockfish_path,
        depth=15
    )

    try:
        stockfish_analysis = (
            stockfish.analyze(
                fen
            )
        )

        print()
        print("=" * 60)
        print("STOCKFISH ANALYSIS")
        print("=" * 60)

        if (
            stockfish_analysis["mate"]
            is not None
        ):
            print(
                f"Evaluation: "
                f"Mate in "
                f"{stockfish_analysis['mate']}"
            )

        elif (
            stockfish_analysis["score"]
            is not None
        ):
            print(
                f"Evaluation: "
                f"{stockfish_analysis['score'] / 100:+.2f}"
            )

        print(
            f"Depth: "
            f"{stockfish_analysis['depth']}"
        )

        print(
            f"Best move (UCI): "
            f"{stockfish_analysis['best_move']}"
        )

        if stockfish_analysis["best_line"]:
            stockfish_san_line = (
                uci_line_to_san(
                    fen,
                    stockfish_analysis[
                        "best_line"
                    ]
                )
            )

            print(
                f"Best line (UCI): "
                f"{' '.join(stockfish_analysis['best_line'])}"
            )

            print(
                f"Best line (SAN): "
                f"{' '.join(stockfish_san_line)}"
            )

        print()
        print("=" * 60)
        print("COMPARISON")
        print("=" * 60)

        same_move = (
            custom_analysis["best_move"]
            == stockfish_analysis["best_move"]
        )

        print(
            f"Custom best move: "
            f"{custom_analysis['best_move']}"
        )

        print(
            f"Stockfish best move: "
            f"{stockfish_analysis['best_move']}"
        )

        print(
            f"Same best move: "
            f"{same_move}"
        )

        if (
            custom_analysis[
                "search_evaluation"
            ]
            is not None
            and stockfish_analysis[
                "score"
            ]
            is not None
        ):
            print(
                f"Custom search evaluation: "
                f"{custom_analysis['search_evaluation'] / 100:+.2f}"
            )

            print(
                f"Stockfish evaluation: "
                f"{stockfish_analysis['score'] / 100:+.2f}"
            )

    finally:
        stockfish.close()


if __name__ == "__main__":
    main()