from ultralytics import YOLO


class PieceDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, image, confidence=0.25):
        results = self.model.predict(
            source=image,
            conf=confidence,
            iou=0.5,
            agnostic_nms=True,
            verbose=False
        )

        detections = []

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence_score = float(box.conf[0])

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detections.append({
                    "class_id": class_id,
                    "class_name": self.model.names[class_id],
                    "confidence": confidence_score,
                    "bbox": (
                        x1,
                        y1,
                        x2,
                        y2
                    )
                })

        return detections

    def draw_detections(self, image, confidence=0.25):
        results = self.model.predict(
            source=image,
            conf=confidence,
            iou=0.5,
            agnostic_nms=True,
            verbose=False
        )

        return results[0].plot()