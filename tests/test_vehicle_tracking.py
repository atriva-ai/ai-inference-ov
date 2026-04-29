#!/usr/bin/env python3
"""
Vehicle Detection and Tracking Test
===================================

This script tests the vehicle detection and tracking model with OpenVINO.
It can process both images and videos, detecting and tracking vehicles with annotated outputs.
Supports both simple IoU tracking and ByteTrack algorithm.

Usage:
    python test_vehicle_tracking.py --input test_images/sample_vehicles.jpg
    python test_vehicle_tracking.py --input test_videos/sample_traffic.mp4 --video
    python test_vehicle_tracking.py --input test_videos/sample_traffic.mp4 --video --tracker bytetrack
"""

import os
import sys
import json
import argparse
import cv2
import numpy as np
import yaml
from pathlib import Path
from openvino import Core
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

console = Console()

# Vehicle-related COCO class IDs for filtering
VEHICLE_CLASS_IDS = {1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
PERSON_CLASS_ID = 0


@dataclass
class STrack:
    """Single object tracking representation for ByteTrack"""
    track_id: int
    bbox: List[float]  # [x, y, w, h]
    score: float
    class_id: int
    class_name: str
    state: str = 'tracked'  # 'tracked', 'lost', 'removed'
    frame_id: int = 0
    start_frame: int = 0
    tracklet_len: int = 0
    
    # Kalman filter state (simplified - just velocity)
    velocity: List[float] = field(default_factory=lambda: [0, 0, 0, 0])
    
    def predict(self):
        """Predict next position based on velocity"""
        self.bbox = [
            self.bbox[0] + self.velocity[0],
            self.bbox[1] + self.velocity[1],
            self.bbox[2] + self.velocity[2],
            self.bbox[3] + self.velocity[3]
        ]
    
    def update(self, new_bbox: List[float], new_score: float, frame_id: int):
        """Update track with new detection"""
        # Update velocity (simple moving average)
        alpha = 0.3
        self.velocity = [
            alpha * (new_bbox[i] - self.bbox[i]) + (1 - alpha) * self.velocity[i]
            for i in range(4)
        ]
        self.bbox = new_bbox
        self.score = new_score
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.state = 'tracked'


class ByteTracker:
    """
    ByteTrack: Multi-Object Tracking by Associating Every Detection Box
    
    Key idea: Use both high and low confidence detections for tracking.
    - First associate high confidence detections with existing tracks
    - Then associate low confidence detections with remaining unmatched tracks
    - This helps maintain tracks during occlusion
    """
    
    def __init__(self, 
                 track_thresh: float = 0.5,      # High confidence threshold
                 track_buffer: int = 30,          # Frames to keep lost tracks
                 match_thresh: float = 0.8,       # IoU threshold for matching
                 low_thresh: float = 0.1):        # Low confidence threshold
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.low_thresh = low_thresh
        
        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.removed_stracks: List[STrack] = []
        
        self.frame_id = 0
        self.next_id = 1
    
    def _get_next_id(self) -> int:
        ret = self.next_id
        self.next_id += 1
        return ret
    
    @staticmethod
    def iou_distance(tracks: List[STrack], detections: List[dict]) -> np.ndarray:
        """Calculate IoU distance matrix between tracks and detections"""
        if len(tracks) == 0 or len(detections) == 0:
            return np.zeros((len(tracks), len(detections)))
        
        cost_matrix = np.zeros((len(tracks), len(detections)))
        
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                # Calculate IoU
                t_box = track.bbox  # [x, y, w, h]
                d_box = det['bbox']  # [x, y, w, h]
                
                # Convert to [x1, y1, x2, y2]
                t_x1, t_y1 = t_box[0], t_box[1]
                t_x2, t_y2 = t_box[0] + t_box[2], t_box[1] + t_box[3]
                d_x1, d_y1 = d_box[0], d_box[1]
                d_x2, d_y2 = d_box[0] + d_box[2], d_box[1] + d_box[3]
                
                # Intersection
                inter_x1 = max(t_x1, d_x1)
                inter_y1 = max(t_y1, d_y1)
                inter_x2 = min(t_x2, d_x2)
                inter_y2 = min(t_y2, d_y2)
                
                if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                    iou = 0.0
                else:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    t_area = t_box[2] * t_box[3]
                    d_area = d_box[2] * d_box[3]
                    union_area = t_area + d_area - inter_area
                    iou = inter_area / union_area if union_area > 0 else 0.0
                
                cost_matrix[i, j] = 1 - iou  # Cost = 1 - IoU
        
        return cost_matrix
    
    @staticmethod
    def linear_assignment(cost_matrix: np.ndarray, thresh: float) -> Tuple[List, List, List]:
        """Simple greedy assignment"""
        if cost_matrix.size == 0:
            return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))
        
        matches = []
        unmatched_tracks = list(range(cost_matrix.shape[0]))
        unmatched_dets = list(range(cost_matrix.shape[1]))
        
        # Greedy matching
        while True:
            if len(unmatched_tracks) == 0 or len(unmatched_dets) == 0:
                break
            
            # Find minimum cost
            min_cost = float('inf')
            min_i, min_j = -1, -1
            
            for i in unmatched_tracks:
                for j in unmatched_dets:
                    if cost_matrix[i, j] < min_cost:
                        min_cost = cost_matrix[i, j]
                        min_i, min_j = i, j
            
            if min_cost > thresh or min_i == -1:
                break
            
            matches.append((min_i, min_j))
            unmatched_tracks.remove(min_i)
            unmatched_dets.remove(min_j)
        
        return matches, unmatched_tracks, unmatched_dets
    
    def update(self, detections: List[dict]) -> Dict[int, dict]:
        """
        Update tracker with new detections
        
        Args:
            detections: List of detection dicts with 'bbox', 'confidence', 'class_id', 'class_name'
        
        Returns:
            Dictionary of active tracks {track_id: track_info}
        """
        self.frame_id += 1
        
        # Separate high and low confidence detections
        high_dets = [d for d in detections if d['confidence'] >= self.track_thresh]
        low_dets = [d for d in detections if self.low_thresh <= d['confidence'] < self.track_thresh]
        
        # Predict new locations for existing tracks
        for track in self.tracked_stracks:
            track.predict()
        
        # ----- First association: high confidence detections with tracked tracks -----
        cost_matrix = self.iou_distance(self.tracked_stracks, high_dets)
        matches, u_tracks, u_dets = self.linear_assignment(cost_matrix, 1 - self.match_thresh)
        
        # Update matched tracks
        for track_idx, det_idx in matches:
            track = self.tracked_stracks[track_idx]
            det = high_dets[det_idx]
            track.update(det['bbox'], det['confidence'], self.frame_id)
        
        # ----- Second association: low confidence detections with unmatched tracks -----
        remaining_tracks = [self.tracked_stracks[i] for i in u_tracks]
        cost_matrix = self.iou_distance(remaining_tracks, low_dets)
        matches2, u_tracks2, _ = self.linear_assignment(cost_matrix, 1 - 0.5)  # Lower IoU threshold
        
        for track_idx, det_idx in matches2:
            track = remaining_tracks[track_idx]
            det = low_dets[det_idx]
            track.update(det['bbox'], det['confidence'], self.frame_id)
            u_tracks.remove(self.tracked_stracks.index(track))
        
        # ----- Handle unmatched tracks -----
        for track_idx in u_tracks:
            track = self.tracked_stracks[track_idx]
            track.state = 'lost'
        
        # Move lost tracks
        new_lost = [self.tracked_stracks[i] for i in u_tracks]
        self.lost_stracks.extend(new_lost)
        
        # ----- Third association: unmatched high dets with lost tracks -----
        remaining_dets = [high_dets[i] for i in u_dets]
        cost_matrix = self.iou_distance(self.lost_stracks, remaining_dets)
        matches3, u_lost, u_dets3 = self.linear_assignment(cost_matrix, 1 - self.match_thresh)
        
        reactivated = []
        for track_idx, det_idx in matches3:
            track = self.lost_stracks[track_idx]
            det = remaining_dets[det_idx]
            track.update(det['bbox'], det['confidence'], self.frame_id)
            reactivated.append(track)
        
        # ----- Create new tracks for remaining unmatched detections -----
        new_tracks = []
        for det_idx in u_dets3:
            det = remaining_dets[det_idx]
            new_track = STrack(
                track_id=self._get_next_id(),
                bbox=det['bbox'],
                score=det['confidence'],
                class_id=det['class_id'],
                class_name=det['class_name'],
                frame_id=self.frame_id,
                start_frame=self.frame_id,
                tracklet_len=1
            )
            new_tracks.append(new_track)
        
        # ----- Update track lists -----
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == 'tracked']
        self.tracked_stracks.extend(reactivated)
        self.tracked_stracks.extend(new_tracks)
        
        # Remove old lost tracks
        self.lost_stracks = [t for t in self.lost_stracks 
                           if t not in reactivated and 
                           self.frame_id - t.frame_id < self.track_buffer]
        
        # Return active tracks as dict
        result = {}
        for track in self.tracked_stracks:
            result[track.track_id] = {
                'bbox': track.bbox,
                'class_name': track.class_name,
                'confidence': track.score,
                'age': track.tracklet_len,
                'class_id': track.class_id
            }
        
        return result

class VehicleTrackingTester:
    """Vehicle Detection and Tracking Tester using YOLOv8n"""
    
    def __init__(self, tracker_type: str = 'iou'):
        """
        Initialize tracker.
        
        Args:
            tracker_type: 'iou' for simple IoU tracking, 'bytetrack' for ByteTrack algorithm
        """
        self.tracker_type = tracker_type
        
        # Use YOLOv8n model for detection (it works!)
        self.models_dir = Path("../models/yolov8n")
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Load model configuration
        with open(self.models_dir / "model.json", 'r') as f:
            self.config = json.load(f)
        
        # Initialize OpenVINO
        self.ie = Core()
        self.model = None
        self.input_shape = self.config["input_shape"][2:]  # [640, 640]
        
        # Load COCO classes from ultralytics metadata
        self.all_classes = self._load_classes_from_metadata()
        if not self.all_classes:
            self.all_classes = self.config.get("classes", [])
        
        # Vehicle classes we track
        self.vehicle_classes = ['bicycle', 'car', 'motorcycle', 'bus', 'truck', 'person']
        
        self.confidence_threshold = self.config.get("confidence_threshold", 0.25)
        self.nms_threshold = self.config.get("nms_threshold", 0.45)
        self.tracking_threshold = 0.5
        self.max_tracks = 50
        
        # Initialize tracker based on type
        if tracker_type == 'bytetrack':
            self.byte_tracker = ByteTracker(
                track_thresh=0.5,
                track_buffer=30,
                match_thresh=0.8,
                low_thresh=0.1
            )
            console.print(f"[green]Using ByteTrack algorithm[/green]")
        else:
            # Simple IoU tracking state
            self.tracks = {}
            self.next_track_id = 1
            console.print(f"[green]Using simple IoU tracking[/green]")
        
        # Generate colors for each track
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(200, 3), dtype=np.uint8)
        
        console.print(f"[green]Vehicle Tracking Tester initialized (using YOLOv8n)[/green]")
        console.print(f"[blue]Input shape: {self.input_shape}[/blue]")
        console.print(f"[blue]Tracking classes: {self.vehicle_classes}[/blue]")
        console.print(f"[blue]Tracker: {tracker_type}[/blue]")
    
    def _load_classes_from_metadata(self):
        """Load class names from ultralytics metadata.yaml"""
        search_paths = [
            self.models_dir / "ultralytics" / "metadata.yaml",
            self.models_dir / "metadata.yaml",
        ]
        
        for metadata_path in search_paths:
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = yaml.safe_load(f)
                    names_dict = metadata.get('names', {})
                    if names_dict and len(names_dict) >= 80:
                        max_idx = max(names_dict.keys())
                        classes = [''] * (max_idx + 1)
                        for idx, name in names_dict.items():
                            classes[idx] = name
                        console.print(f"[blue]Loaded {len(classes)} classes from {metadata_path}[/blue]")
                        return classes
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not load metadata: {e}[/yellow]")
        return []
    
    def load_model(self):
        """Load YOLOv8n OpenVINO model for vehicle detection"""
        xml_path = self.models_dir / "yolov8n.xml"
        bin_path = self.models_dir / "yolov8n.bin"
        
        if not xml_path.exists() or not bin_path.exists():
            console.print("[red]YOLOv8n model files not found. Please run test_runner.py --download-models first.[/red]")
            return False
        
        try:
            model = self.ie.read_model(model=xml_path)
            self.model = self.ie.compile_model(model=model, device_name="CPU")
            console.print("[green]✅ YOLOv8n model loaded for vehicle tracking[/green]")
            return True
        except Exception as e:
            console.print(f"[red]❌ Failed to load model: {str(e)}[/red]")
            return False
    
    def preprocess_image(self, image):
        """Preprocess image for YOLOv8 inference"""
        input_h, input_w = self.input_shape
        
        # Resize image to model input size
        resized_image = cv2.resize(image, (input_w, input_h))
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized_image = rgb_image.astype(np.float32) / 255.0
        
        # Transpose to CHW format
        input_data = np.transpose(normalized_image, (2, 0, 1))
        input_data = np.expand_dims(input_data, axis=0)
        
        return input_data
    
    def postprocess_detections(self, outputs, original_shape):
        """Post-process YOLOv8 output and filter for vehicles/persons"""
        detections = []
        
        # YOLOv8 output shape is (1, 84, 8400) -> transpose to (8400, 84)
        predictions = outputs[0].T  # Shape: (8400, 84)
        
        orig_h, orig_w = original_shape[:2]
        input_h, input_w = self.input_shape
        scale_x = orig_w / input_w
        scale_y = orig_h / input_h
        
        boxes = []
        confidences = []
        class_ids = []
        
        for pred in predictions:
            cx, cy, w, h = pred[:4]
            class_scores = pred[4:]
            
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]
            
            # Filter by confidence and vehicle/person classes
            if confidence > self.confidence_threshold:
                class_name = self.all_classes[class_id] if class_id < len(self.all_classes) else f"class_{class_id}"
                
                # Only keep vehicles and persons
                if class_name in self.vehicle_classes:
                    x1 = int((cx - w / 2) * scale_x)
                    y1 = int((cy - h / 2) * scale_y)
                    box_w = int(w * scale_x)
                    box_h = int(h * scale_y)
                    
                    boxes.append([x1, y1, box_w, box_h])
                    confidences.append(float(confidence))
                    class_ids.append(int(class_id))
        
        # Apply NMS
        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
            
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    class_id = class_ids[i]
                    class_name = self.all_classes[class_id] if class_id < len(self.all_classes) else f"class_{class_id}"
                    
                    detections.append({
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': confidences[i],
                        'bbox': [x, y, w, h]  # x, y, w, h format
                    })
        
        return detections
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        x1_1, y1_1, w1, h1 = box1
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        
        x1_2, y1_2, w2, h2 = box2
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def update_tracks(self, detections):
        """Update vehicle tracks based on new detections using selected tracker"""
        if self.tracker_type == 'bytetrack':
            return self.byte_tracker.update(detections)
        else:
            return self._update_tracks_iou(detections)
    
    def _update_tracks_iou(self, detections):
        """Simple IoU-based tracking algorithm"""
        matched_detections = set()
        updated_tracks = {}
        
        # Match detections to existing tracks
        for track_id, track in self.tracks.items():
            best_iou = 0
            best_detection_idx = -1
            
            for i, detection in enumerate(detections):
                if i in matched_detections:
                    continue
                
                iou = self.calculate_iou(track['bbox'], detection['bbox'])
                if iou > best_iou and iou > self.tracking_threshold:
                    best_iou = iou
                    best_detection_idx = i
            
            if best_detection_idx != -1:
                # Update existing track
                detection = detections[best_detection_idx]
                updated_tracks[track_id] = {
                    'bbox': detection['bbox'],
                    'class_name': detection['class_name'],
                    'confidence': detection['confidence'],
                    'age': track['age'] + 1,
                    'last_seen': 0
                }
                matched_detections.add(best_detection_idx)
            else:
                # Track not matched, increment age
                track['age'] += 1
                track['last_seen'] += 1
                
                # Remove old tracks
                if track['last_seen'] < 5:  # Keep tracks for 5 frames
                    updated_tracks[track_id] = track
        
        # Create new tracks for unmatched detections
        for i, detection in enumerate(detections):
            if i not in matched_detections and len(updated_tracks) < self.max_tracks:
                track_id = self.next_track_id
                self.next_track_id += 1
                
                updated_tracks[track_id] = {
                    'bbox': detection['bbox'],
                    'class_name': detection['class_name'],
                    'confidence': detection['confidence'],
                    'age': 1,
                    'last_seen': 0
                }
        
        self.tracks = updated_tracks
        return self.tracks
    
    def draw_tracks(self, image, tracks):
        """Draw vehicle tracks on image with unique colors per track"""
        annotated_image = image.copy()
        
        for track_id, track in tracks.items():
            bbox = track['bbox']
            class_name = track['class_name']
            confidence = track['confidence']
            age = track['age']
            
            # bbox is now [x, y, w, h] in pixel coordinates
            x, y, w, h = bbox
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
            
            # Ensure coordinates are within image bounds
            img_h, img_w = image.shape[:2]
            x1 = max(0, min(x1, img_w - 1))
            y1 = max(0, min(y1, img_h - 1))
            x2 = max(0, min(x2, img_w - 1))
            y2 = max(0, min(y2, img_h - 1))
            
            # Use unique color per track ID for consistent tracking visualization
            color = tuple(map(int, self.colors[track_id % len(self.colors)]))
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw track ID and info
            label = f"ID:{track_id} {class_name} ({confidence:.2f})"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_image, (x1, y1 - label_h - 10), 
                         (x1 + label_w, y1), color, -1)
            cv2.putText(annotated_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated_image
    
    def test_image(self, input_path, output_path):
        """Test vehicle detection and tracking on a single image"""
        console.print(f"[blue]Testing Vehicle Tracking on image: {input_path}[/blue]")
        
        # Load image
        image = cv2.imread(input_path)
        if image is None:
            console.print(f"[red]❌ Could not load image: {input_path}[/red]")
            return False
        
        # Preprocess image
        input_data = self.preprocess_image(image)
        
        # Run inference
        start_time = time.time()
        output_tensor = self.model.output(0)
        result = self.model(input_data)
        outputs = result[output_tensor]
        
        inference_time = time.time() - start_time
        
        # Post-process detections
        detections = self.postprocess_detections(outputs, image.shape)
        
        # Update tracks
        tracks = self.update_tracks(detections)
        
        # Draw tracks
        annotated_image = self.draw_tracks(image, tracks)
        
        # Save result
        cv2.imwrite(output_path, annotated_image)
        
        console.print(f"[green]✅ Inference completed in {inference_time:.3f}s[/green]")
        console.print(f"[green]✅ Found {len(detections)} vehicles[/green]")
        console.print(f"[green]✅ Active tracks: {len(tracks)}[/green]")
        console.print(f"[green]✅ Saved annotated image: {output_path}[/green]")
        
        # Print detection details
        for i, detection in enumerate(detections):
            console.print(f"[cyan]  {i+1}. {detection['class_name']}: {detection['confidence']:.3f}[/cyan]")
        
        return True
    
    def test_video(self, input_path, output_path):
        """Test vehicle detection and tracking on a video"""
        console.print(f"[blue]Testing Vehicle Tracking on video: {input_path}[/blue]")
        
        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            console.print(f"[red]❌ Could not open video: {input_path}[/red]")
            return False
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        console.print(f"[blue]Video properties: {width}x{height}, {fps} FPS, {total_frames} frames[/blue]")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        total_detections = 0
        max_tracks = 0
        total_inference_time = 0.0
        total_processing_time = 0.0
        
        processing_start = time.time()
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Processing video...", total=total_frames)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_start = time.time()
                
                # Preprocess frame
                input_data = self.preprocess_image(frame)
                
                # Run inference (measure time)
                inference_start = time.time()
                output_tensor = self.model.output(0)
                result = self.model(input_data)
                outputs = result[output_tensor]
                inference_time = time.time() - inference_start
                total_inference_time += inference_time
                
                # Post-process detections
                detections = self.postprocess_detections(outputs, frame.shape)
                
                # Update tracks
                tracks = self.update_tracks(detections)
                
                # Draw tracks
                annotated_frame = self.draw_tracks(frame, tracks)
                
                # Write frame
                out.write(annotated_frame)
                
                frame_count += 1
                total_detections += len(detections)
                max_tracks = max(max_tracks, len(tracks))
                progress.update(task, advance=1)
        
        total_processing_time = time.time() - processing_start
        
        cap.release()
        out.release()
        
        # Calculate FPS metrics
        avg_inference_time = total_inference_time / frame_count if frame_count > 0 else 0
        inference_fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
        overall_fps = frame_count / total_processing_time if total_processing_time > 0 else 0
        
        console.print(f"\n[bold cyan]═══ Summary Report ═══[/bold cyan]")
        console.print(f"[green]✅ Processed {frame_count} frames[/green]")
        console.print(f"[green]✅ Total processing time: {total_processing_time:.2f}s[/green]")
        console.print(f"[green]✅ Overall FPS (including I/O): {overall_fps:.2f}[/green]")
        console.print(f"[green]✅ Inference FPS (model only): {inference_fps:.2f}[/green]")
        console.print(f"[green]✅ Avg inference time per frame: {avg_inference_time*1000:.2f}ms[/green]")
        console.print(f"[green]✅ Total detections: {total_detections}[/green]")
        console.print(f"[green]✅ Average detections per frame: {total_detections/frame_count:.2f}[/green]")
        console.print(f"[green]✅ Maximum active tracks: {max_tracks}[/green]")
        console.print(f"[green]✅ Tracker: {self.tracker_type}[/green]")
        console.print(f"[green]✅ Saved annotated video: {output_path}[/green]")
        
        return True

def main():
    parser = argparse.ArgumentParser(description="Vehicle Detection and Tracking Test")
    parser.add_argument("--input", type=str, required=True, help="Input image or video path")
    parser.add_argument("--output", type=str, help="Output path")
    parser.add_argument("--video", action="store_true", help="Process as video")
    parser.add_argument("--tracker", choices=['iou', 'bytetrack'], default='iou',
                       help="Tracking algorithm: 'iou' (simple IoU matching) or 'bytetrack' (ByteTrack algorithm)")
    
    args = parser.parse_args()
    
    # Initialize tester with selected tracker
    tester = VehicleTrackingTester(tracker_type=args.tracker)
    
    # Load model
    if not tester.load_model():
        return
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        if args.video or args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            output_path = "output/vehicle_tracking_video_output.mp4"
        else:
            output_path = "output/vehicle_tracking_image_output.jpg"
    
    # Run test
    if args.video or args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        success = tester.test_video(args.input, output_path)
    else:
        success = tester.test_image(args.input, output_path)
    
    if success:
        console.print("[green]🎉 Vehicle tracking test completed successfully![/green]")
    else:
        console.print("[red]❌ Vehicle tracking test failed![/red]")

if __name__ == "__main__":
    main()
