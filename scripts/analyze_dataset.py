from collections import Counter
from pathlib import Path


CLASS_NAMES = {
    0: "white_pawn",
    1: "white_knight",
    2: "white_bishop",
    3: "white_rook",
    4: "white_queen",
    5: "white_king",
    6: "black_pawn",
    7: "black_knight",
    8: "black_bishop",
    9: "black_rook",
    10: "black_queen",
    11: "black_king"
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png"
}


def get_images(directory):
    return [
        file
        for file in directory.iterdir()
        if file.suffix.lower() in IMAGE_EXTENSIONS
    ]


def analyze_split(dataset_root, split):
    images_directory = (
        dataset_root
        / "images"
        / split
    )

    labels_directory = (
        dataset_root
        / "labels"
        / split
    )

    images = get_images(images_directory)
    labels = list(labels_directory.glob("*.txt"))

    image_names = {
        image.stem
        for image in images
    }

    label_names = {
        label.stem
        for label in labels
    }

    missing_labels = (
        image_names
        - label_names
    )

    missing_images = (
        label_names
        - image_names
    )

    class_counts = Counter()

    empty_labels = []
    invalid_labels = []

    total_annotations = 0

    for label_file in labels:
        with open(
            label_file,
            "r",
            encoding="utf-8"
        ) as file:
            lines = [
                line.strip()
                for line in file
                if line.strip()
            ]

        if not lines:
            empty_labels.append(
                label_file.name
            )

        for line_number, line in enumerate(
            lines,
            start=1
        ):
            parts = line.split()

            if len(parts) != 5:
                invalid_labels.append(
                    (
                        label_file.name,
                        line_number,
                        "Incorrect number of values"
                    )
                )
                continue

            try:
                class_id = int(parts[0])

                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

            except ValueError:
                invalid_labels.append(
                    (
                        label_file.name,
                        line_number,
                        "Invalid numeric value"
                    )
                )
                continue

            if class_id not in CLASS_NAMES:
                invalid_labels.append(
                    (
                        label_file.name,
                        line_number,
                        f"Unknown class: {class_id}"
                    )
                )
                continue

            values = [
                x_center,
                y_center,
                width,
                height
            ]

            if not all(
                0 <= value <= 1
                for value in values
            ):
                invalid_labels.append(
                    (
                        label_file.name,
                        line_number,
                        "Coordinates outside 0-1 range"
                    )
                )
                continue

            if width == 0 or height == 0:
                invalid_labels.append(
                    (
                        label_file.name,
                        line_number,
                        "Bounding box has zero size"
                    )
                )
                continue

            class_counts[class_id] += 1
            total_annotations += 1

    return {
        "images": len(images),
        "labels": len(labels),
        "annotations": total_annotations,
        "missing_labels": missing_labels,
        "missing_images": missing_images,
        "empty_labels": empty_labels,
        "invalid_labels": invalid_labels,
        "class_counts": class_counts
    }


def print_split_report(split, report):
    print()
    print("=" * 50)
    print(split.upper())
    print("=" * 50)

    print(
        f"Images: {report['images']}"
    )

    print(
        f"Label files: {report['labels']}"
    )

    print(
        f"Annotations: {report['annotations']}"
    )

    if report["images"] > 0:
        average = (
            report["annotations"]
            / report["images"]
        )

        print(
            f"Average annotations per image: "
            f"{average:.2f}"
        )

    print()
    print("Class distribution:")

    for class_id, class_name in CLASS_NAMES.items():
        count = report["class_counts"].get(
            class_id,
            0
        )

        print(
            f"{class_id:2} "
            f"{class_name:15} "
            f"{count}"
        )

    print()

    print(
        "Missing labels:",
        len(report["missing_labels"])
    )

    print(
        "Missing images:",
        len(report["missing_images"])
    )

    print(
        "Empty labels:",
        len(report["empty_labels"])
    )

    print(
        "Invalid labels:",
        len(report["invalid_labels"])
    )

    if report["missing_labels"]:
        print()
        print("Images without labels:")

        for name in sorted(
            report["missing_labels"]
        ):
            print(name)

    if report["missing_images"]:
        print()
        print("Labels without images:")

        for name in sorted(
            report["missing_images"]
        ):
            print(name)

    if report["empty_labels"]:
        print()
        print("Empty label files:")

        for name in report["empty_labels"]:
            print(name)

    if report["invalid_labels"]:
        print()
        print("Invalid annotations:")

        for (
            filename,
            line_number,
            reason
        ) in report["invalid_labels"]:

            print(
                f"{filename}:"
                f"{line_number} - "
                f"{reason}"
            )


def main():
    project_root = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    dataset_root = (
        project_root
        / "data"
        / "piece_dataset"
    )

    splits = [
        "train",
        "val",
        "test"
    ]

    total_classes = Counter()

    total_images = 0
    total_annotations = 0

    for split in splits:
        report = analyze_split(
            dataset_root,
            split
        )

        print_split_report(
            split,
            report
        )

        total_images += report["images"]

        total_annotations += (
            report["annotations"]
        )

        total_classes.update(
            report["class_counts"]
        )

    print()
    print("=" * 50)
    print("TOTAL")
    print("=" * 50)

    print(
        f"Images: {total_images}"
    )

    print(
        f"Annotations: {total_annotations}"
    )

    print()
    print("Total class distribution:")

    for class_id, class_name in CLASS_NAMES.items():
        print(
            f"{class_id:2} "
            f"{class_name:15} "
            f"{total_classes[class_id]}"
        )


if __name__ == "__main__":
    main()