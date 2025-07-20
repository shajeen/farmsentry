"""
Enhanced Animal Detection with Multi-Object Support
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import Tuple, Dict, List

class AnimalDetector:
    def __init__(self, config_module):
        self.config = config_module
        print("Loading YOLOv8 model...")
        self.model = YOLO('models/yolov8m.pt')  # Medium model for better accuracy
        self.model.to(self.config.DEVICE)
        self.class_names = self.model.names
        print(f"Model loaded on {self.config.DEVICE.upper()}")

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, List[Tuple[int, int, int, int, float]]]]:
        """Process frame with enhanced multi-animal detection"""
        # Run detection with optimized parameters
        results = self.model(
            frame,
            imgsz=self.config.DETECTION_SIZE,
            conf=self.config.CONFIDENCE_THRESHOLD,
            iou=self.config.IOU_THRESHOLD,
            device=self.config.DEVICE
        )[0]

        detections: Dict[str, List[Tuple[int, int, int, int, float]]] = {}
        annotated_frame = frame.copy()

        for box in results.boxes:
            cls_id = int(box.cls)
            confidence = float(box.conf.item())
            class_name = self.class_names[cls_id]

            # Skip omitted classes
            if class_name in self.config.OMIT_CLASSES:
                continue

            # Get bounding box coordinates
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
            area = (x_max - x_min) * (y_max - y_min)

            # Skip small detections
            if area < self.config.MIN_ANIMAL_AREA:
                continue

            # Skip overlapping detections
            if self._is_overlapping((x_min, y_min, x_max, y_max), detections):
                continue

            # Store detection
            if class_name not in detections:
                detections[class_name] = []
            detections[class_name].append((x_min, y_min, x_max, y_max, confidence))

            # Draw bounding box
            color = (0, 0, 255)  # Red
            cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), color, 2)
            
            # Draw label with confidence
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(annotated_frame, label, (x_min, y_min-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return annotated_frame, detections

    def _calculate_iou(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Calculate union area
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - intersection_area

        return intersection_area / union_area if union_area > 0 else 0.0

    def _is_overlapping(self, new_box: Tuple[int, int, int, int],
                       existing_detections: Dict[str, List[Tuple[int, int, int, int, float]]]) -> bool:
        """Check if a new detection significantly overlaps with existing ones."""
        for boxes in existing_detections.values():
            for existing_box in boxes:
                iou = self._calculate_iou(new_box, existing_box[:4])
                if iou > self.config.IOU_THRESHOLD:
                    return True
        return False