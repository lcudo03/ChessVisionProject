from pathlib import Path

import torch
from ultralytics import YOLO


def main():
    project_root = Path(__file__).resolve().parent.parent

    dataset_yaml = (
        project_root
        / "data"
        / "piece_dataset"
        / "chess_pieces.yaml"
    )

    print("Project root:", project_root)
    print("Dataset:", dataset_yaml)
    print("Dataset exists:", dataset_yaml.exists())

    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        device = 0
    else:
        print("Training on CPU")
        device = "cpu"

    model = YOLO("../src/piece_detection/yolo26n.pt")

    model.train(
        data=str(dataset_yaml),
        epochs=50,
        imgsz=640,
        batch=8,
        device=device,
        project=str(project_root / "runs"),
        name="chess_pieces_baseline"
    )


if __name__ == "__main__":
    main()