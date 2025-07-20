import cv2
from ultralytics import YOLO
import numpy as np
from typing import List, Tuple, Dict, Any
import yaml
import os

class YOLOConeDetector:
    def __init__(self, config_path: str = 'config/model_config.yaml', device: str = 'cpu'):
        """
        Initialize the YOLOv8 cone detector using config.
        Args:
            config_path (str): Path to the model config YAML.
            device (str): 'cpu' or 'cuda'.
        """
        with open(config_path, 'r') as f:
            model_cfg = yaml.safe_load(f)
        cone_cfg = model_cfg['cone_detection']
        self.model_path = cone_cfg['model']
        self.confidence = cone_cfg['confidence']
        self.iou_threshold = cone_cfg['iou_threshold']
        self.device = device
        self.model = YOLO(self.model_path)

    def detect_cones(self, image: np.ndarray, use_color_filtering: bool = True) -> List[Dict[str, Any]]:
        """
        Run cone detection on a single image/frame.
        Args:
            image (np.ndarray): Input image (BGR, as from OpenCV).
            use_color_filtering (bool): Whether to use color filtering to identify orange objects as cones.
        Returns:
            List of cone detections, each as a dict with keys: 'bbox', 'confidence', 'class_id', 'class_name'.
        """
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.model(img_rgb, conf=self.confidence, iou=self.iou_threshold, device=self.device)
        detections = []
        
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes, 'xyxy') else []
            scores = r.boxes.conf.cpu().numpy() if hasattr(r.boxes, 'conf') else []
            class_ids = r.boxes.cls.cpu().numpy() if hasattr(r.boxes, 'cls') else []
            
            for box, score, class_id in zip(boxes, scores, class_ids):
                class_name = self.model.names[int(class_id)] if hasattr(self.model, 'names') else str(class_id)
                
                # Check if detection could be a cone
                is_cone = False
                
                # Direct cone detection
                if 'cone' in class_name.lower():
                    is_cone = True
                
                # Use color filtering for development with base model
                elif use_color_filtering and self._is_orange_object(image, box):
                    # Look for objects that could be cones based on class type and color
                    if any(keyword in class_name.lower() for keyword in ['person', 'bottle', 'cup', 'vase', 'chair']):
                        is_cone = True
                        class_name = f"cone_candidate_{class_name}"
                
                if is_cone:
                    detections.append({
                        'bbox': box.tolist(),
                        'confidence': float(score),
                        'class_id': int(class_id),
                        'class_name': class_name
                    })
        
        return detections
    
    def _is_orange_object(self, image: np.ndarray, bbox: List[float]) -> bool:
        """
        Check if the detected object is predominantly orange (cone color).
        Args:
            image (np.ndarray): Input image (BGR).
            bbox (list): Bounding box [x1, y1, x2, y2].
        Returns:
            bool: True if object appears to be orange.
        """
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return False
        
        # Extract region of interest
        roi = image[y1:y2, x1:x2]
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Define orange color range in HSV
        lower_orange1 = np.array([5, 100, 100])   # Lower orange
        upper_orange1 = np.array([15, 255, 255])
        lower_orange2 = np.array([160, 100, 100]) # Upper orange (wraps around)
        upper_orange2 = np.array([180, 255, 255])
        
        # Create masks for orange colors
        mask1 = cv2.inRange(hsv, lower_orange1, upper_orange1)
        mask2 = cv2.inRange(hsv, lower_orange2, upper_orange2)
        orange_mask = cv2.bitwise_or(mask1, mask2)
        
        # Calculate percentage of orange pixels
        orange_pixels = cv2.countNonZero(orange_mask)
        total_pixels = roi.shape[0] * roi.shape[1]
        
        if total_pixels == 0:
            return False
        
        orange_percentage = orange_pixels / total_pixels
        
        # Consider it orange if at least 20% of pixels are orange
        return orange_percentage > 0.2 