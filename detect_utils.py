import cv2
import numpy as np
import torch
from typing import Tuple, Dict, Any, List
import base64
import os
from io import BytesIO
from PIL import Image
import pathlib

# Fix for Windows path issue
posix = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

class BallTracker:
    def __init__(self, model_path: str = 'best.pt', conf_threshold: float = 0.5):
        """Initialize YOLOv5 model for golf ball detection."""
        # Convert model path to absolute path
        model_path = os.path.abspath(model_path)
        
        # Load model with explicit source
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                  path=model_path, 
                                  force_reload=True,
                                  skip_validation=True)
        self.model.conf = conf_threshold
        self.model.eval()
        self.previous_positions = []
        self.max_history = 5
        self.center_threshold = 0.1  # 10% of frame width from center

    def process_frame(self, frame: np.ndarray) -> Tuple[str, str]:
        """
        Process a single frame to detect golf ball and determine direction.
        
        Args:
            frame: Input frame in BGR format
            
        Returns:
            tuple: (base64_encoded_image, direction_label)
        """
        # Make a copy for drawing
        frame_copy = frame.copy()
        h, w = frame.shape[:2]
        center_x = w // 2
        
        # Draw center line
        cv2.line(frame_copy, (center_x, 0), (center_x, h), (0, 255, 0), 2)
        
        # Run detection
        results = self.model(frame_copy)
        detections = results.xyxy[0].cpu().numpy()
        
        direction = "Center"
        ball_detected = False
        
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            if conf > self.model.conf:
                ball_detected = True
                # Draw bounding box
                cv2.rectangle(frame_copy, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                
                # Calculate center of the detection
                ball_center_x = (x1 + x2) / 2
                
                # Update position history
                self.previous_positions.append(ball_center_x)
                if len(self.previous_positions) > self.max_history:
                    self.previous_positions.pop(0)
                
                # Determine direction based on position relative to center
                rel_pos = (ball_center_x - center_x) / w
                if abs(rel_pos) < self.center_threshold:
                    direction = "Center"
                elif rel_pos < 0:
                    direction = "Left"
                else:
                    direction = "Right"
                
                # Draw direction indicator
                cv2.putText(frame_copy, f"Direction: {direction}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                break
        
        # Convert frame to base64
        _, buffer = cv2.imencode('.jpg', frame_copy)
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        return img_str, direction if ball_detected else "No ball detected"