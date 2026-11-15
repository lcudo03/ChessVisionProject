from pathlib import Path
import shutil


CLASS_MAPPING = {
    0: 0,   # white-pawn
    1: 3,   # white-rook
    2: 1,   # white-knight
    3: 2,   # white-bishop
    4: 4,   # white-queen
    5: 5,   # white-king

    6: 6,   # black-pawn
    7: 9,   # black-rook
    8: 7,   # black-knight
    9: 8,   # black-bishop
    10: 10, # black-queen
    11: 11  # black-king
}

EMPTY_CLASS_ID = 12

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png"
}


def convert_label(source_label, destination_label):
    converted_lines = []

    with open(source_label, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()

            if not parts:
                continue

            old_class_id = int(parts[0])

            if old_class_id == EMPTY_CLASS_ID:
                continue

            if old_class_id not in CLASS_MAPPING:
                raise ValueError(
                    f"Unknown class id {old_class_id} "
                    f"in {source_label}"
                )

            new_class_id = CLASS_MAPPING[old_class_id]

            parts[0] = str(new_class_id)

            converted_lines.append(
                " ".join(parts)
            )

    with open(
        destination_label,
        "w",
        encoding="utf-8"
    ) as file:
        for line in converted_lines:
            file.write(line + "\n")


def merge_split(
    source_root,
    destination_root,
    split
):
    source_images = (
        source_root
        / split
        / "images"
    )

    source_labels = (
        source_root
        / split
        / "labels"
    )

    destination_images = (
        destination_root
        / "images"
        / split
    )

    destination_labels = (
        destination_root
        / "labels"
        / split
    )

    destination_images.mkdir(
        parents=True,
        exist_ok=True
    )

    destination_labels.mkdir(
        parents=True,
        exist_ok=True
    )

    image_files = [
        file
        for file in source_images.iterdir()
        if file.suffix.lower() in IMAGE_EXTENSIONS
    ]

    copied_images = 0
    converted_labels = 0
    missing_labels = 0

    for image_file in image_files:
        source_label = (
            source_labels
            / f"{image_file.stem}.txt"
        )

        if not source_label.exists():
            print(
                f"Missing label: {image_file.name}"
            )

            missing_labels += 1
            continue

        new_stem = (
            f"chessred2k_{image_file.stem}"
        )

        destination_image = (
            destination_images
            / f"{new_stem}{image_file.suffix.lower()}"
        )

        destination_label = (
            destination_labels
            / f"{new_stem}.txt"
        )

        shutil.copy2(
            image_file,
            destination_image
        )

        convert_label(
            source_label,
            destination_label
        )

        copied_images += 1
        converted_labels += 1

    print()
    print(split.upper())
    print(
        f"Images copied: {copied_images}"
    )
    print(
        f"Labels converted: {converted_labels}"
    )
    print(
        f"Missing labels: {missing_labels}"
    )


def main():
    project_root = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    source_root = (
        project_root
        / "data"
        / "piece_dataset2"
        / "yolo_dataset"
    )

    destination_root = (
        project_root
        / "data"
        / "piece_dataset"
    )

    print(
        "Source:",
        source_root
    )

    print(
        "Destination:",
        destination_root
    )

    for split in [
        "train",
        "val",
        "test"
    ]:
        merge_split(
            source_root,
            destination_root,
            split
        )

    print()
    print(
        "ChessReD2K merge finished"
    )


if __name__ == "__main__":
    main()