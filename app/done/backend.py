import cv2
import numpy as np
from ultralytics import YOLO
import onnxruntime as ort
import os
import json

class PlantDetector:
    def __init__(self):
        self.yolo_model = None
        self.classify_session = None
        self.labels = []
        self.display_names = {} # Khởi tạo thêm biến này
        self.conf_threshold = 0.5
        self.img_size = 224
        self.load_models()

    def load_models(self):
        # Load YOLO
        yolo_path = "model/best.pt"
        if not os.path.exists(yolo_path):
            raise FileNotFoundError("Không tìm thấy model/best.pt")
        self.yolo_model = YOLO(yolo_path)

        # Load classifier ONNX
        onnx_path = "model/plant_model.onnx"
        if not os.path.exists(onnx_path):
            raise FileNotFoundError("Không tìm thấy model/plant_model.onnx")

        self.classify_session = ort.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.classify_session.get_inputs()[0].name

        # Load model_config.json
        config_path = "model/model_config.json"
        if not os.path.exists(config_path):
            raise FileNotFoundError("Không tìm thấy model/model_config.json")

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        self.labels = cfg["known_classes"]
        # Lấy thêm dictionary tên hiển thị
        self.display_names = cfg.get("display_names", {}) 
        self.conf_threshold = float(cfg["confidence_threshold"])
        self.img_size = int(cfg["img_size"])

    def detect_trees(self, image):
        results = self.yolo_model.predict(source=image, conf=0.25, iou=0.5, imgsz=640)
        detected = []
        for r in results:
            if r.masks is None:
                continue
            for i, (box, mask) in enumerate(zip(r.boxes.xyxy, r.masks.xy)):
                x1, y1, x2, y2 = map(int, box)
                area = (x2 - x1) * (y2 - y1)
                detected.append({
                    'box': (x1, y1, x2, y2),
                    'mask': mask,
                    'area': area,
                    'index': i + 1
                })
        return detected

    def classify_crop(self, crop):
        if crop.size == 0:
            return "Unknown", 0.0

        crop = cv2.resize(crop, (self.img_size, self.img_size))
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = crop.astype(np.float32) / 255.0
        crop = np.expand_dims(crop, axis=0)

        preds = self.classify_session.run(None, {self.input_name: crop})[0]
        class_id = int(np.argmax(preds))
        max_prob = float(preds[0][class_id])

        if max_prob < self.conf_threshold:
            return "Unknown", max_prob * 100

        # Lấy label từ known_classes
        internal_label = self.labels[class_id]
        # Chuyển sang display_name tương ứng
        display_name = self.display_names.get(internal_label, internal_label)

        return display_name, max_prob * 100

    def draw_results(self, image, detections, predictions=None):
        img = image.copy()
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['box']
            mask = det['mask']
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            pts = mask.reshape((-1, 1, 2)).astype(int)
            cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
            cv2.putText(img, str(det['index']),
                        (x1, y1 - 15), cv2.FONT_HERSHEY_DUPLEX,
                        0.9, (0, 255, 255), 2)

            if predictions and i < len(predictions):
                label, prob = predictions[i]
                cv2.putText(img, f"{label} ({prob:.1f}%)",
                            (x1, y1 - 40), cv2.FONT_HERSHEY_DUPLEX,
                            0.8, (255, 255, 0), 2)
        return img
