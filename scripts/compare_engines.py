from pathlib import Path
import sys
import csv
import time
import statistics

import chess


project_root = Path(
    __file__
).resolve().parent.parent

src_path = (
    project_root
    / "src"
)

sys.path.insert(
    0,
    str(src_path)
)


from chess_logic.evaluator import CustomEvaluator
from chess_engine.stockfish_engine import StockfishAnalyzer


input_path = (
    project_root
    / "evaluation"
    / "engine_test_positions.csv"
)

output_path = (
    project_root
    / "evaluation"
    / "engine_comparison.csv"
)

stockfish_path = (
    project_root
    / "engine"
    / "stockfish.exe"
)


def calculate_loss(
    stockfish_best_result,
    custom_move_result
):
    if (
        stockfish_best_result is None
        or custom_move_result is None
    ):
        return None

    best_score = stockfish_best_result[
        "score_for_side_to_move"
    ]

    custom_score = custom_move_result[
        "score_for_side_to_move"
    ]

    if (
        best_score is None
        or custom_score is None
    ):
        return None

    loss = (
        best_score
        - custom_score
    )

    return max(
        0,
        loss
    )


def get_mate_status(
    stockfish_best_result,
    custom_move_result
):
    if (
        stockfish_best_result is None
        or custom_move_result is None
    ):
        return ""

    best_mate = stockfish_best_result[
        "mate_for_side_to_move"
    ]

    custom_mate = custom_move_result[
        "mate_for_side_to_move"
    ]

    if (
        best_mate is None
        and custom_mate is None
    ):
        return ""

    if (
        best_mate is not None
        and best_mate > 0
    ):
        if (
            custom_mate is not None
            and custom_mate > 0
        ):
            return "keeps_forced_mate"

        return "misses_forced_mate"

    if (
        custom_mate is not None
        and custom_mate < 0
        and (
            best_mate is None
            or best_mate >= 0
        )
    ):
        return "allows_forced_mate"

    return "mate_related"


def classify_move(
    same_move,
    loss_cp,
    mate_status
):
    if same_move:
        return "best_move"

    if mate_status == "keeps_forced_mate":
        return "mating_alternative"

    if mate_status == "misses_forced_mate":
        return "missed_mate"

    if mate_status == "allows_forced_mate":
        return "allows_mate"

    if mate_status == "mate_related":
        return "mate_related"

    if loss_cp is None:
        return "unknown"

    if loss_cp <= 20:
        return "near_best"

    if loss_cp <= 50:
        return "good"

    if loss_cp <= 100:
        return "inaccuracy"

    if loss_cp <= 200:
        return "mistake"

    return "large_error"


def format_evaluation(score):
    if score is None:
        return "N/A"

    return f"{score / 100:+.2f}"


def main():
    custom_engine = CustomEvaluator()

    stockfish = StockfishAnalyzer(
        engine_path=stockfish_path,
        depth=15
    )

    results = []

    try:
        with open(
            input_path,
            newline="",
            encoding="utf-8"
        ) as file:
            reader = csv.DictReader(
                file
            )

            for index, row in enumerate(
                reader,
                start=1
            ):
                fen = row["fen"]

                board = chess.Board(
                    fen
                )

                print()
                print("=" * 70)
                print(
                    f"Position {index}"
                )
                print(
                    f"FEN: {fen}"
                )

                custom_start = (
                    time.perf_counter()
                )

                custom_result = (
                    custom_engine.analyze(
                        fen
                    )
                )

                custom_time = (
                    time.perf_counter()
                    - custom_start
                )

                stockfish_start = (
                    time.perf_counter()
                )

                stockfish_result = (
                    stockfish.analyze(
                        fen
                    )
                )

                stockfish_time = (
                    time.perf_counter()
                    - stockfish_start
                )

                custom_move = (
                    custom_result[
                        "best_move"
                    ]
                )

                stockfish_move = (
                    stockfish_result[
                        "best_move"
                    ]
                )

                same_move = (
                    custom_move
                    == stockfish_move
                )

                custom_score = (
                    custom_result[
                        "search_evaluation"
                    ]
                )

                stockfish_score = (
                    stockfish_result[
                        "score"
                    ]
                )

                stockfish_best_forced_result = None
                custom_move_stockfish_result = None

                forced_best_analysis_time = 0
                forced_custom_analysis_time = 0

                evaluation_loss_cp = None
                mate_status = ""
                move_quality = "unknown"

                if stockfish_move is not None:
                    forced_best_start = (
                        time.perf_counter()
                    )

                    stockfish_best_forced_result = (
                        stockfish.evaluate_move(
                            fen,
                            stockfish_move
                        )
                    )

                    forced_best_analysis_time = (
                        time.perf_counter()
                        - forced_best_start
                    )

                if custom_move is not None:
                    if (
                        custom_move
                        == stockfish_move
                    ):
                        custom_move_stockfish_result = (
                            stockfish_best_forced_result
                        )

                        evaluation_loss_cp = 0

                    else:
                        forced_custom_start = (
                            time.perf_counter()
                        )

                        custom_move_stockfish_result = (
                            stockfish.evaluate_move(
                                fen,
                                custom_move
                            )
                        )

                        forced_custom_analysis_time = (
                            time.perf_counter()
                            - forced_custom_start
                        )

                        evaluation_loss_cp = (
                            calculate_loss(
                                stockfish_best_forced_result,
                                custom_move_stockfish_result
                            )
                        )

                    mate_status = (
                        get_mate_status(
                            stockfish_best_forced_result,
                            custom_move_stockfish_result
                        )
                    )

                    move_quality = (
                        classify_move(
                            same_move,
                            evaluation_loss_cp,
                            mate_status
                        )
                    )

                stockfish_forced_best_score = None
                stockfish_forced_custom_score = None

                if (
                    stockfish_best_forced_result
                    is not None
                ):
                    stockfish_forced_best_score = (
                        stockfish_best_forced_result[
                            "score"
                        ]
                    )

                if (
                    custom_move_stockfish_result
                    is not None
                ):
                    stockfish_forced_custom_score = (
                        custom_move_stockfish_result[
                            "score"
                        ]
                    )

                print(
                    f"Side to move: "
                    f"{'White' if board.turn else 'Black'}"
                )

                print(
                    f"Custom move: "
                    f"{custom_move}"
                )

                print(
                    f"Stockfish move: "
                    f"{stockfish_move}"
                )

                print(
                    f"Same move: "
                    f"{same_move}"
                )

                print()

                if custom_score is not None:
                    print(
                        f"Custom evaluation: "
                        f"{custom_score / 100:+.2f}"
                    )

                if stockfish_score is not None:
                    print(
                        f"Stockfish initial evaluation: "
                        f"{stockfish_score / 100:+.2f}"
                    )

                if (
                    stockfish_forced_best_score
                    is not None
                ):
                    print(
                        "Stockfish evaluation "
                        "of best move: "
                        f"{format_evaluation(
                            stockfish_forced_best_score
                        )}"
                    )

                if (
                    stockfish_forced_custom_score
                    is not None
                ):
                    print(
                        "Stockfish evaluation "
                        "of custom move: "
                        f"{format_evaluation(
                            stockfish_forced_custom_score
                        )}"
                    )

                if evaluation_loss_cp is not None:
                    print(
                        f"Evaluation loss: "
                        f"{evaluation_loss_cp} cp "
                        f"({evaluation_loss_cp / 100:.2f})"
                    )

                print(
                    f"Move quality: "
                    f"{move_quality}"
                )

                if mate_status:
                    print(
                        f"Mate status: "
                        f"{mate_status}"
                    )

                if (
                    stockfish_result[
                        "mate"
                    ]
                    is not None
                ):
                    print(
                        f"Stockfish mate: "
                        f"{stockfish_result['mate']}"
                    )

                print()

                print(
                    f"Custom time: "
                    f"{custom_time:.4f} s"
                )

                print(
                    f"Stockfish initial time: "
                    f"{stockfish_time:.4f} s"
                )

                print(
                    f"Stockfish forced best time: "
                    f"{forced_best_analysis_time:.4f} s"
                )

                print(
                    f"Stockfish forced custom time: "
                    f"{forced_custom_analysis_time:.4f} s"
                )

                stockfish_best_mate = None
                stockfish_custom_move_mate = None

                if (
                    stockfish_best_forced_result
                    is not None
                ):
                    stockfish_best_mate = (
                        stockfish_best_forced_result[
                            "mate_for_side_to_move"
                        ]
                    )

                if (
                    custom_move_stockfish_result
                    is not None
                ):
                    stockfish_custom_move_mate = (
                        custom_move_stockfish_result[
                            "mate_for_side_to_move"
                        ]
                    )

                results.append(
                    {
                        "position": index,

                        "fen": fen,

                        "side_to_move":
                            "white"
                            if board.turn
                            else "black",

                        "custom_best_move":
                            custom_move,

                        "stockfish_best_move":
                            stockfish_move,

                        "same_best_move":
                            same_move,

                        "custom_evaluation_cp":
                            custom_score,

                        "stockfish_initial_evaluation_cp":
                            stockfish_score,

                        "stockfish_best_move_evaluation_cp":
                            stockfish_forced_best_score,

                        "stockfish_custom_move_evaluation_cp":
                            stockfish_forced_custom_score,

                        "evaluation_loss_cp":
                            evaluation_loss_cp,

                        "move_quality":
                            move_quality,

                        "stockfish_best_mate":
                            stockfish_best_mate,

                        "stockfish_custom_move_mate":
                            stockfish_custom_move_mate,

                        "mate_status":
                            mate_status,

                        "custom_time_s":
                            custom_time,

                        "stockfish_initial_time_s":
                            stockfish_time,

                        "stockfish_forced_best_time_s":
                            forced_best_analysis_time,

                        "stockfish_forced_custom_time_s":
                            forced_custom_analysis_time
                    }
                )

    finally:
        stockfish.close()

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        fieldnames = [
            "position",
            "fen",
            "side_to_move",

            "custom_best_move",
            "stockfish_best_move",
            "same_best_move",

            "custom_evaluation_cp",

            "stockfish_initial_evaluation_cp",

            "stockfish_best_move_evaluation_cp",

            "stockfish_custom_move_evaluation_cp",

            "evaluation_loss_cp",
            "move_quality",

            "stockfish_best_mate",
            "stockfish_custom_move_mate",
            "mate_status",

            "custom_time_s",

            "stockfish_initial_time_s",

            "stockfish_forced_best_time_s",

            "stockfish_forced_custom_time_s"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    total_positions = len(
        results
    )

    same_moves = sum(
        1
        for result in results
        if result[
            "same_best_move"
        ]
    )

    normal_losses = [
        result[
            "evaluation_loss_cp"
        ]
        for result in results
        if (
            result[
                "evaluation_loss_cp"
            ]
            is not None
            and not result[
                "mate_status"
            ]
        )
    ]

    near_best = sum(
        1
        for loss in normal_losses
        if loss <= 20
    )

    within_50 = sum(
        1
        for loss in normal_losses
        if loss <= 50
    )

    within_100 = sum(
        1
        for loss in normal_losses
        if loss <= 100
    )

    missed_mates = sum(
        1
        for result in results
        if result[
            "mate_status"
        ]
        == "misses_forced_mate"
    )

    allows_mates = sum(
        1
        for result in results
        if result[
            "mate_status"
        ]
        == "allows_forced_mate"
    )

    mating_alternatives = sum(
        1
        for result in results
        if result[
            "mate_status"
        ]
        == "keeps_forced_mate"
    )

    if total_positions > 0:
        move_agreement = (
            same_moves
            / total_positions
        )

        average_custom_time = (
            sum(
                result[
                    "custom_time_s"
                ]
                for result in results
            )
            / total_positions
        )

        average_stockfish_time = (
            sum(
                result[
                    "stockfish_initial_time_s"
                ]
                for result in results
            )
            / total_positions
        )
    else:
        move_agreement = 0
        average_custom_time = 0
        average_stockfish_time = 0

    print()
    print("=" * 70)
    print(
        "ENGINE COMPARISON SUMMARY"
    )
    print("=" * 70)

    print(
        f"Positions: "
        f"{total_positions}"
    )

    print(
        f"Exact same best move: "
        f"{same_moves}/{total_positions} "
        f"({move_agreement:.2%})"
    )

    if normal_losses:
        average_loss = (
            sum(
                normal_losses
            )
            / len(
                normal_losses
            )
        )

        median_loss = (
            statistics.median(
                normal_losses
            )
        )

        print()
        print(
            "NON-MATE POSITION QUALITY"
        )

        print(
            f"Positions evaluated: "
            f"{len(normal_losses)}"
        )

        print(
            f"Average loss: "
            f"{average_loss:.2f} cp "
            f"({average_loss / 100:.2f})"
        )

        print(
            f"Median loss: "
            f"{median_loss:.2f} cp "
            f"({median_loss / 100:.2f})"
        )

        print(
            f"Within 20 cp: "
            f"{near_best}/"
            f"{len(normal_losses)} "
            f"({near_best / len(normal_losses):.2%})"
        )

        print(
            f"Within 50 cp: "
            f"{within_50}/"
            f"{len(normal_losses)} "
            f"({within_50 / len(normal_losses):.2%})"
        )

        print(
            f"Within 100 cp: "
            f"{within_100}/"
            f"{len(normal_losses)} "
            f"({within_100 / len(normal_losses):.2%})"
        )

    print()
    print(
        "MATE POSITIONS"
    )

    print(
        f"Mating alternatives: "
        f"{mating_alternatives}"
    )

    print(
        f"Missed forced mates: "
        f"{missed_mates}"
    )

    print(
        f"Moves allowing forced mate: "
        f"{allows_mates}"
    )

    print()
    print(
        "PERFORMANCE"
    )

    print(
        f"Average custom time: "
        f"{average_custom_time:.4f} s"
    )

    print(
        f"Average Stockfish time: "
        f"{average_stockfish_time:.4f} s"
    )

    print()
    print(
        f"Results saved to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()