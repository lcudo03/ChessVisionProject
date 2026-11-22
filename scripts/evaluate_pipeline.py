from pathlib import Path
import sys
import csv

import cv2


project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"

sys.path.insert(0, str(src_path))

from pipeline import ChessVisionPipeline


evaluation_dir = project_root / "evaluation"
images_dir = evaluation_dir / "images"

ground_truth_path = (
    evaluation_dir
    / "ground_truth.csv"
)

results_path = (
    evaluation_dir
    / "results.csv"
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


def fen_to_board(fen):
    position = fen.split()[0]

    board = []

    for fen_row in position.split("/"):
        row = []

        for char in fen_row:
            if char.isdigit():
                row.extend(
                    [None] * int(char)
                )
            else:
                row.append(char)

        board.append(row)

    return board


def calculate_square_accuracy(
    predicted_board,
    expected_board
):
    correct = 0

    for row in range(8):
        for col in range(8):
            if (
                predicted_board[row][col]
                == expected_board[row][col]
            ):
                correct += 1

    return correct / 64


def calculate_piece_accuracy(
    predicted_board,
    expected_board
):
    expected_pieces = 0
    correct_pieces = 0

    for row in range(8):
        for col in range(8):
            expected = expected_board[row][col]
            predicted = predicted_board[row][col]

            if expected is not None:
                expected_pieces += 1

                if predicted == expected:
                    correct_pieces += 1

    if expected_pieces == 0:
        return 1.0

    return (
        correct_pieces
        / expected_pieces
    )


def main():
    pipeline = ChessVisionPipeline(
        piece_model_path=piece_model_path,
        corner_model_path=corner_model_path
    )

    results = []

    total_images = 0
    successful_images = 0
    failed_images = 0
    exact_fen_count = 0

    square_accuracies = []
    piece_accuracies = []

    with open(
        ground_truth_path,
        newline="",
        encoding="utf-8"
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            image_name = row["image"]
            expected_fen = row["fen"]

            total_images += 1

            image_path = (
                images_dir
                / image_name
            )

            image = cv2.imread(
                str(image_path)
            )

            print()
            print("=" * 70)
            print(
                f"Evaluating: {image_name}"
            )

            if image is None:
                print("Could not load image")

                failed_images += 1

                results.append(
                    {
                        "image": image_name,
                        "status": "FAILED",
                        "square_accuracy": "",
                        "piece_accuracy": "",
                        "exact_fen": False,
                        "predicted_fen": "",
                        "expected_fen": expected_fen,
                        "error": "Could not load image"
                    }
                )

                continue

            try:
                result = pipeline.recognize(
                    image
                )

                predicted_fen = (
                    result["fen"]
                )

                predicted_board = (
                    result["board"]
                )

                expected_board = (
                    fen_to_board(
                        expected_fen
                    )
                )

                square_accuracy = (
                    calculate_square_accuracy(
                        predicted_board,
                        expected_board
                    )
                )

                piece_accuracy = (
                    calculate_piece_accuracy(
                        predicted_board,
                        expected_board
                    )
                )

                exact_fen = (
                    predicted_fen
                    == expected_fen
                )

                successful_images += 1

                square_accuracies.append(
                    square_accuracy
                )

                piece_accuracies.append(
                    piece_accuracy
                )

                if exact_fen:
                    exact_fen_count += 1

                results.append(
                    {
                        "image": image_name,
                        "status": "OK",
                        "square_accuracy":
                            square_accuracy,
                        "piece_accuracy":
                            piece_accuracy,
                        "exact_fen":
                            exact_fen,
                        "predicted_fen":
                            predicted_fen,
                        "expected_fen":
                            expected_fen,
                        "error": ""
                    }
                )

                print(
                    f"Square accuracy: "
                    f"{square_accuracy:.2%}"
                )

                print(
                    f"Piece accuracy: "
                    f"{piece_accuracy:.2%}"
                )

                print(
                    f"Exact FEN: "
                    f"{exact_fen}"
                )

            except Exception as error:
                failed_images += 1

                print(
                    f"Pipeline failed: "
                    f"{error}"
                )

                results.append(
                    {
                        "image": image_name,
                        "status": "FAILED",
                        "square_accuracy": "",
                        "piece_accuracy": "",
                        "exact_fen": False,
                        "predicted_fen": "",
                        "expected_fen": expected_fen,
                        "error": str(error)
                    }
                )

    with open(
        results_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        fieldnames = [
            "image",
            "status",
            "square_accuracy",
            "piece_accuracy",
            "exact_fen",
            "predicted_fen",
            "expected_fen",
            "error"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    if successful_images > 0:
        average_square_accuracy = (
            sum(square_accuracies)
            / successful_images
        )

        average_piece_accuracy = (
            sum(piece_accuracies)
            / successful_images
        )

        exact_fen_successful = (
            exact_fen_count
            / successful_images
        )
    else:
        average_square_accuracy = 0.0
        average_piece_accuracy = 0.0
        exact_fen_successful = 0.0

    if total_images > 0:
        localization_success = (
            successful_images
            / total_images
        )

        overall_exact_fen = (
            exact_fen_count
            / total_images
        )
    else:
        localization_success = 0.0
        overall_exact_fen = 0.0

    print()
    print("=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    print(
        f"Total images: "
        f"{total_images}"
    )

    print(
        f"Successful pipeline runs: "
        f"{successful_images}/{total_images} "
        f"({localization_success:.2%})"
    )

    print(
        f"Pipeline failures: "
        f"{failed_images}/{total_images} "
        f"({failed_images / total_images:.2%})"
    )

    print()

    print(
        f"Average square accuracy: "
        f"{average_square_accuracy:.2%}"
    )

    print(
        f"Average piece accuracy: "
        f"{average_piece_accuracy:.2%}"
    )

    print(
        f"Exact FEN among successful runs: "
        f"{exact_fen_count}/{successful_images} "
        f"({exact_fen_successful:.2%})"
    )

    print(
        f"Overall exact FEN accuracy: "
        f"{exact_fen_count}/{total_images} "
        f"({overall_exact_fen:.2%})"
    )

    print()
    print(
        f"Results saved to: "
        f"{results_path}"
    )


if __name__ == "__main__":
    main()