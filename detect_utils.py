import cv2
import numpy as np
import torch
from typing import Tuple, Optional
import base64
import os

class BallTracker:
    def __init__(self, model_path: str, conf_threshold: float = 0.5):
        """
        Initialize YOLOv5 model for ball detection.
        
        Args:
            model_path: Path to the YOLOv5 .pt model file
            conf_threshold: Confidence threshold for detections (0-1)
        """
        try:
            # Load model
            self.model = torch.hub.load(
                'ultralytics/yolov5', 
                'custom',
                path=model_path,
                force_reload=True,
                skip_validation=True
            )
            self.model.conf = conf_threshold
            self.model.eval()
            self.center_threshold = 0.1  # 10% of frame width from center
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model: {str(e)}")

    def _get_direction(self, bbox: np.ndarray, frame_width: int) -> str:
        """Determine direction based on ball position."""
        x1, y1, x2, y2, conf, cls = bbox
        center_x = frame_width / 2
        ball_center = (x1 + x2) / 2
        rel_pos = (ball_center - center_x) / frame_width
        
        if abs(rel_pos) < self.center_threshold:
            return "Center"
        return "Left" if rel_pos < 0 else "Right"

    def process_frame(self, frame: np.ndarray) -> Tuple[str, str]:
        """
        Process a single frame to detect ball and determine direction.

        Args:
            frame: Input frame in BGR format

        Returns:
            tuple: (base64_encoded_image, direction)
        """
        if frame is None or frame.size == 0:
            raise ValueError("Invalid frame input")
            
        frame_copy = frame.copy()
        h, w = frame.shape[:2]
        center_x = w // 2
        
        # Draw center line
        cv2.line(frame_copy, (center_x, 0), (center_x, h), (0, 255, 0), 2)
        
        # Run detection
        with torch.no_grad():
            results = self.model(frame_copy)
        
        detections = results.xyxy[0].cpu().numpy()
        direction = "No ball detected"
        
        # Process detections
        for det in detections:
            if det[4] > self.model.conf:  # Check confidence
                x1, y1, x2, y2 = map(int, det[:4])
                
                # Draw bounding box
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Get and display direction
                direction = self._get_direction(det, w)
                cv2.putText(
                    frame_copy, 
                    f"Direction: {direction}", 
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 255, 0), 
                    2
                )
                break  # Process only the highest confidence detection
        
        # Encode result as base64
        _, buffer = cv2.imencode('.jpg', frame_copy, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        return img_str, direction
