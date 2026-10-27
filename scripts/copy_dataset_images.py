from pathlib import Path
import shutil


SPLITS = {
    "train": "train",
    "valid": "val",
    "test": "test"
}


def copy_images(source_root, destination_root):
    for source_split, destination_split in SPLITS.items():
        source_images = (
            source_root
            / source_split
            / "images"
        )

        destination_images = (
            destination_root
            / "images"
            / destination_split
        )

        destination_labels = (
            destination_root
            / "labels"
            / destination_split
        )

        destination_images.mkdir(
            parents=True,
            exist_ok=True
        )

        if not source_images.exists():
            print(
                f"Images directory not found: {source_images}"
            )
            continue

        image_files = []

        for extension in ("*.jpg", "*.jpeg", "*.png"):
            image_files.extend(
                source_images.glob(extension)
            )

        print(
            f"{source_split}: "
            f"{len(image_files)} images found"
        )

        copied = 0
        missing_labels = 0

        for image_file in image_files:
            label_file = (
                destination_labels
                / f"{image_file.stem}.txt"
            )

            if not label_file.exists():
                print(
                    f"Missing label for: {image_file.name}"
                )

                missing_labels += 1
                continue

            destination_file = (
                destination_images
                / image_file.name
            )

            shutil.copy2(
                image_file,
                destination_file
            )

            copied += 1

        print(
            f"Copied: {copied}"
        )

        print(
            f"Skipped because of missing labels: "
            f"{missing_labels}"
        )

        print()


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

    copy_images(
        source_root,
        destination_root
    )

    print("Image copying finished")