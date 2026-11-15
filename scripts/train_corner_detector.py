from pathlib import Path

import torch
from ultralytics import YOLO


def main():
    project_root = Path(__file__).resolve().parent.parent

    dataset_yaml = (
        project_root
        / "data"
        / "corner_dataset"
        / "yolo_corner_dataset"
        / "dataset.yaml"
    )

    if not dataset_yaml.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {dataset_yaml}"
        )

    device = 0 if torch.cuda.is_available() else "cpu"

    print("CUDA available:", torch.cuda.is_available())
    print("Device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    model = YOLO("yolo26n.pt")

    model.train(
        data=str(dataset_yaml),
        epochs=50,
        imgsz=640,
        batch=8,
        device=device,
        workers=0,
        cache=False,
        project=str(project_root / "runs"),
        name="chess_corner_detector",
        seed=0
    )


if __name__ == "__main__":
    main()