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

    def detect(self, image, confidence=0.05):
        results = self.model.predict(
            source=image,
            conf=confidence,
            iou=0.5,
            verbose=False
        )

        result = results[0]

        all_detections = []

        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence_score = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            class_name = self.model.names[class_id]

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            all_detections.append(
                {
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
            )

        best_detections = {}

        for corner_name in self.REQUIRED_CORNERS:
            candidates = [
                detection
                for detection in all_detections
                if detection["class_name"] == corner_name
            ]

            if not candidates:
                continue

            best_detection = max(
                candidates,
                key=lambda detection: detection["confidence"]
            )

            best_detections[corner_name] = best_detection

        missing = [
            corner
            for corner in self.REQUIRED_CORNERS
            if corner not in best_detections
        ]

        if (
            missing == ["bottom_right"]
            and "top_left" in best_detections
            and "top_right" in best_detections
            and "bottom_left" in best_detections
        ):
            top_left_candidates = [
                detection
                for detection in all_detections
                if detection["class_name"] == "top_left"
            ]

            if len(top_left_candidates) >= 2:
                top_left_candidates = sorted(
                    top_left_candidates,
                    key=lambda detection: detection["confidence"],
                    reverse=True
                )[:2]

                candidate_a = top_left_candidates[0]
                candidate_b = top_left_candidates[1]

                top_right = best_detections[
                    "top_right"
                ]["center"]

                bottom_left = best_detections[
                    "bottom_left"
                ]["center"]

                def signed_area(
                    top_left,
                    top_right,
                    bottom_right,
                    bottom_left
                ):
                    points = [
                        top_left,
                        top_right,
                        bottom_right,
                        bottom_left
                    ]

                    area = 0.0

                    for i in range(4):
                        x1, y1 = points[i]
                        x2, y2 = points[(i + 1) % 4]

                        area += (
                            x1 * y2
                            - x2 * y1
                        )

                    return area / 2.0

                area_a = signed_area(
                    candidate_a["center"],
                    top_right,
                    candidate_b["center"],
                    bottom_left
                )

                area_b = signed_area(
                    candidate_b["center"],
                    top_right,
                    candidate_a["center"],
                    bottom_left
                )

                if area_a > 0:
                    correct_top_left = candidate_a
                    recovered_bottom_right = candidate_b

                elif area_b > 0:
                    correct_top_left = candidate_b
                    recovered_bottom_right = candidate_a

                else:
                    raise ValueError(
                        "Could not geometrically recover "
                        "bottom_right corner"
                    )

                best_detections["top_left"] = correct_top_left

                best_detections["bottom_right"] = {
                    **recovered_bottom_right,
                    "class_name": "bottom_right"
                }

                print(
                    "Recovered bottom_right "
                    "using corner geometry"
                )

        missing = [
            corner
            for corner in self.REQUIRED_CORNERS
            if corner not in best_detections
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