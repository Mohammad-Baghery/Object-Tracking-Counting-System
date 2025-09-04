
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import torch
from ultralytics import YOLO
from deep_sort_realtime import DeepSort
import config


class ObjectTracker:
    """Object detection and tracking using YOLOv8 + DeepSORT"""

    def __init__(self, model_path: str = None, device: str = "auto"):
        """
        Initialize object tracker

        Args:
            model_path: Path to YOLOv8 model
            device: Device to run inference on ("auto", "cpu", "cuda", "mps")
        """
        self.model_path = model_path or config.YOLO_MODEL
        self.device = self._setup_device(device)

        # Initialize YOLO model
        print(f"🚀 Loading YOLOv8 model: {self.model_path}")
        self.yolo_model = YOLO(self.model_path)
        self.yolo_model.to(self.device)

        # Initialize DeepSORT tracker
        print("🔄 Initializing DeepSORT tracker...")
        self.deep_sort = DeepSort(
            max_age=config.TRACKER_CONFIG['max_age'],
            n_init=config.TRACKER_CONFIG['n_init'],
            nms_max_overlap=config.TRACKER_CONFIG['nms_max_overlap'],
            max_cosine_distance=config.TRACKER_CONFIG['max_cosine_distance'],
            nn_budget=config.TRACKER_CONFIG['nn_budget'],
        )

        # Configuration
        self.confidence_threshold = config.CONFIDENCE_THRESHOLD
        self.track_classes = config.TRACK_CLASSES
        self.class_names = config.COCO_CLASSES

        print("✅ Object tracker initialized successfully!")

    def _setup_device(self, device: str) -> str:
        """
        Setup computation device

        Args:
            device: Requested device

        Returns:
            Available device
        """
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        print(f"🖥️  Using device: {device}")
        return device

    def detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect objects in frame using YOLOv8

        Args:
            frame: Input frame

        Returns:
            List of detection dictionaries
        """
        # Run YOLO inference
        results = self.yolo_model(frame, verbose=False, conf=self.confidence_threshold)

        detections = []

        for result in results:
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    # Extract detection data
                    xyxy = box.xyxy[0].cpu().numpy()  # Bounding box coordinates
                    conf = box.conf[0].cpu().numpy()  # Confidence score
                    cls = int(box.cls[0].cpu().numpy())  # Class ID

                    # Filter by class and confidence
                    if cls in self.track_classes and conf >= self.confidence_threshold:
                        detection = {
                            'bbox': xyxy.astype(int),  # [x1, y1, x2, y2]
                            'confidence': float(conf),
                            'class_id': cls,
                            'class_name': self.class_names[cls]
                        }
                        detections.append(detection)

        return detections

    def track_objects(self, frame: np.ndarray, detections: List[Dict]) -> List[Dict]:
        """
        Track detected objects using DeepSORT

        Args:
            frame: Input frame
            detections: List of detection dictionaries

        Returns:
            List of tracking results
        """
        if not detections:
            # Update tracker with empty detections to handle disappearing tracks
            self.deep_sort.update_tracks([], frame=frame)
            return []

        # Prepare detections for DeepSORT
        detection_list = []

        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']

            # Convert bbox format from [x1, y1, x2, y2] to [x1, y1, w, h]
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1

            # Create detection tuple: ([x, y, w, h], confidence, class_name)
            detection_list.append(([x1, y1, w, h], confidence, class_name))

        # Update DeepSORT tracker
        tracks = self.deep_sort.update_tracks(detection_list, frame=frame)

        # Process tracking results
        tracking_results = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            bbox = track.to_ltrb()  # Get bounding box in [x1, y1, x2, y2] format
            class_name = track.get_det_class()
            confidence = track.get_det_conf()

            if confidence is None:
                confidence = 0.0

            tracking_result = {
                'track_id': track_id,
                'bbox': bbox.astype(int),
                'class_name': class_name,
                'confidence': float(confidence)
            }

            tracking_results.append(tracking_result)

        return tracking_results

    def process_frame(self, frame: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        """
        Process single frame: detect and track objects

        Args:
            frame: Input frame

        Returns:
            Tuple of (tracking results, processed frame)
        """
        # Detect objects
        detections = self.detect_objects(frame)

        # Track objects
        tracking_results = self.track_objects(frame, detections)

        return tracking_results, frame

    def get_model_info(self) -> Dict[str, any]:
        """
        Get information about loaded models

        Returns:
            Dictionary with model information
        """
        return {
            'yolo_model': self.model_path,
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'track_classes': self.track_classes,
            'class_names': [self.class_names[i] for i in self.track_classes]
        }

    def set_confidence_threshold(self, threshold: float):
        """
        Update confidence threshold

        Args:
            threshold: New confidence threshold (0.0 to 1.0)
        """
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        print(f"🎯 Updated confidence threshold: {self.confidence_threshold}")

    def set_track_classes(self, class_ids: List[int]):
        """
        Update classes to track

        Args:
            class_ids: List of COCO class IDs to track
        """
        valid_classes = [cls for cls in class_ids if 0 <= cls < len(self.class_names)]
        self.track_classes = valid_classes
        print(f"🎯 Updated track classes: {[self.class_names[i] for i in valid_classes]}")
