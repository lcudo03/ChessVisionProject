from pathlib import Path


CLASS_MAPPING = {
    0: 8,
    1: 11,
    2: 7,
    3: 6,
    4: 10,
    5: 9,
    6: 2,
    7: 5,
    8: 1,
    9: 0,
    10: 4,
    11: 3
}


def convert_label_file(source_file, destination_file):
    converted_lines = []

    with open(source_file, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()

            if not parts:
                continue

            old_class_id = int(parts[0])

            if old_class_id not in CLASS_MAPPING:
                raise ValueError(
                    f"Unknown class id {old_class_id} in {source_file}"
                )

            new_class_id = CLASS_MAPPING[old_class_id]

            parts[0] = str(new_class_id)

            converted_lines.append(" ".join(parts))

    destination_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(destination_file, "w", encoding="utf-8") as file:
        for line in converted_lines:
            file.write(line + "\n")


def convert_dataset(source_root, destination_root):
    splits = {
        "train": "train",
        "valid": "val",
        "test": "test"
    }

    for source_split, destination_split in splits.items():
        source_labels = (
            source_root
            / source_split
            / "labels"
        )

        destination_labels = (
            destination_root
            / "labels"
            / destination_split
        )

        if not source_labels.exists():
            print(
                f"Labels directory not found: {source_labels}"
            )
            continue

        label_files = list(
            source_labels.glob("*.txt")
        )

        print(
            f"{source_split}: "
            f"{len(label_files)} label files"
        )

        for source_file in label_files:
            destination_file = (
                destination_labels
                / source_file.name
            )

            convert_label_file(
                source_file,
                destination_file
            )


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent

    source_root = (
        project_root
        / "downloaded_dataset"
    )

    destination_root = (
        project_root
        / "data"
        / "piece_dataset"
    )

    convert_dataset(
        source_root,
        destination_root
    )

    print("Label conversion finished")