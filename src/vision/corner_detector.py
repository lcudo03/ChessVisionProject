from ultralytics import YOLO
import numpy as np


class CornerDetector:
    REQUIRED_CORNERS = [
        "top_left",
        "top_right",
        "bottom_right",
        "bottom_left"
    ]

    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, image, confidence=0.20):
        results = self.model.predict(
            source=image,
            conf=confidence,
            iou=0.5,
            verbose=False
        )

        result = results[0]

        best_detections = {}

        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence_score = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            class_name = self.model.names[class_id]

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            detection = {
                "class_name": class_name,
                "confidence": confidence_score,
                "center": (
                    center_x,
                    center_y
                ),
                "bbox": (
                    x1,
                    y1,
                    x2,
                    y2
                )
            }

            if (
                class_name not in best_detections
                or confidence_score
                > best_detections[class_name]["confidence"]
            ):
                best_detections[class_name] = detection

        missing = [
            name
            for name in self.REQUIRED_CORNERS
            if name not in best_detections
        ]

        if missing:
            raise ValueError(
                f"Missing board corners: {missing}"
            )

        return best_detections

    def get_corner_points(self, detections):
        return np.array(
            [
                detections["top_left"]["center"],
                detections["top_right"]["center"],
                detections["bottom_right"]["center"],
                detections["bottom_left"]["center"]
            ],
            dtype=np.float32
        )