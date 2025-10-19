from ultralytics import YOLO

class Detector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect_from_image(self, image):
        results = self.model(image)
        detected = []
        for r in results[0].boxes:
            x1, y1, x2, y2 = r.xyxy[0]
            w = x2 - x1
            h = y2 - y1
            detected.append({
                "bbox": (int(x1), int(y1), int(w), int(h)),
                "area": int(w * h),
                "class": int(r.cls)  # Class ID
            })
        return {
            "detected": detected,
            "total_area": image.shape[0] * image.shape[1]
        }
