import cv2
from ultralytics import YOLO
import numpy as np
from typing import List, Tuple, Dict, Any

class YOLODetector:
    def __init__(self, model_path: str, confidence: float = 0.5, iou_threshold: float = 0.45, device: str = 'cpu'):
        """
        Initialize the YOLOv8 detector.
        Args:
            model_path (str): Path to the YOLOv8 model file (.pt).
            confidence (float): Confidence threshold for detections.
            iou_threshold (float): IoU threshold for NMS.
            device (str): 'cpu' or 'cuda'.
        """
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.device = device

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run detection on a single image/frame.
        Args:
            image (np.ndarray): Input image (BGR, as from OpenCV).
        Returns:
            List of detections, each as a dict with keys: 'bbox', 'confidence', 'class_id', 'class_name'.
        """
        # YOLO expects RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.model(img_rgb, conf=self.confidence, iou=self.iou_threshold, device=self.device)
        detections = []
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes, 'xyxy') else []
            scores = r.boxes.conf.cpu().numpy() if hasattr(r.boxes, 'conf') else []
            class_ids = r.boxes.cls.cpu().numpy() if hasattr(r.boxes, 'cls') else []
            for box, score, class_id in zip(boxes, scores, class_ids):
                detections.append({
                    'bbox': box.tolist(),  # [x1, y1, x2, y2]
                    'confidence': float(score),
                    'class_id': int(class_id),
                    'class_name': self.model.names[int(class_id)] if hasattr(self.model, 'names') else str(class_id)
                })
        return detections 