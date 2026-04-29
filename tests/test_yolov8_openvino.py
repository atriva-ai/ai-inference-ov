#!/usr/bin/env python3
"""
YOLOv8 Object Detection Test
============================

This script tests YOLOv8 object detection models (n/s/m) with OpenVINO.
It can process both images and videos, detecting objects and creating annotated outputs.

Usage:
    python test_yolov8_openvino.py --input test_images/sample.jpg
    python test_yolov8_openvino.py --input test_images/sample.jpg --size s
    python test_yolov8_openvino.py --input test_videos/sample.mp4 --video --size m
    python test_yolov8_openvino.py --input test_videos/sample.mp4 --video --fps 30
    python test_yolov8_openvino.py --input test_videos/sample.mp4 --video --length 10.5
    python test_yolov8_openvino.py --input test_videos/sample.mp4 --video --inference-fps 1 --length 30
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


console = Console()

class YOLOv8Tester:
    """YOLOv8 Object Detection Tester"""
    
    def __init__(self, model_size='n'):
        self.model_size = model_size
        self.models_dir = Path(f"../models/yolov8{model_size}")
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Load model configuration
        config_path = self.models_dir / "model.json"
        if not config_path.exists():
            console.print(f"[red]❌ Model config not found: {config_path}[/red]")
            sys.exit(1)
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize OpenVINO
        self.ie = Core()
        self.model = None
        self.input_shape = self.config["input_shape"][2:]  # [640, 640]
        
        # Load classes from ultralytics metadata.yaml (COCO 80 classes)
        self.classes = self._load_classes_from_metadata()
        if not self.classes:
            self.classes = self.config.get("classes", [])
        
        self.confidence_threshold = self.config.get("confidence_threshold", 0.25)
        self.nms_threshold = self.config.get("nms_threshold", 0.45)
        
        # Generate unique colors for each class
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(len(self.classes), 3), dtype=np.uint8)
        
        console.print(f"[green]YOLOv8{model_size.upper()} Tester initialized[/green]")
        console.print(f"[blue]Input shape: {self.input_shape}[/blue]")
        console.print(f"[blue]Classes: {len(self.classes)}[/blue]")
        console.print(f"[blue]Confidence threshold: {self.confidence_threshold}[/blue]")
    
    def _load_classes_from_metadata(self):
        """Load class names from ultralytics metadata.yaml (COCO classes)"""
        # Prioritize ultralytics subdirectory (has correct COCO 80 classes)
        search_paths = [
            self.models_dir / "ultralytics" / "metadata.yaml",
            self.models_dir / f"yolov8{self.model_size}_openvino_model" / "metadata.yaml",
            self.models_dir / "metadata.yaml",
        ]
        
        for metadata_path in search_paths:
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = yaml.safe_load(f)
                    
                    names_dict = metadata.get('names', {})
                    if names_dict and len(names_dict) >= 80:  # Ensure it's COCO (80 classes)
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
        """Load YOLOv8 OpenVINO model"""
        xml_path = self.models_dir / f"yolov8{self.model_size}.xml"
        bin_path = self.models_dir / f"yolov8{self.model_size}.bin"
        
        if not xml_path.exists() or not bin_path.exists():
            console.print("[red]Model files not found. Please download the model first.[/red]")
            return False
        
        try:
            model = self.ie.read_model(model=xml_path)
            self.model = self.ie.compile_model(model=model, device_name="CPU")
            console.print("[green]✅ YOLOv8 model loaded successfully[/green]")
            return True
        except Exception as e:
            console.print(f"[red]❌ Failed to load model: {str(e)}[/red]")
            return False
    
    def preprocess_image(self, image):
        """Preprocess image for YOLOv8n inference using letterbox"""
        # Use letterbox resizing to preserve aspect ratio
        resized_image, self.ratio, self.pad = YOLOv8Tester.letterbox(image, new_shape=self.input_shape)
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized_image = rgb_image.astype(np.float32) / 255.0
        
        # Transpose to CHW format
        input_data = np.transpose(normalized_image, (2, 0, 1))
        input_data = np.expand_dims(input_data, axis=0)
        
        return input_data
    
    @staticmethod
    def letterbox(img, new_shape=(640,640), color=(114,114,114)):
        # returns resized_img, ratio, (dw, dh) pad
        # used to preserve aspect ratio and compute inverse mapping
        h0, w0 = img.shape[:2]
        new_w, new_h = new_shape
        r = min(new_w / w0, new_h / h0)
        resized_w, resized_h = int(round(w0 * r)), int(round(h0 * r))
        resized = cv2.resize(img, (resized_w, resized_h))
        dw = new_w - resized_w
        dh = new_h - resized_h
        top = dh // 2
        bottom = dh - top
        left = dw // 2
        right = dw - left
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return padded, r, (left, top)

    def postprocess_detections(self, outputs, original_shape):
        """Postprocess YOLOv8 raw output format (1, 84, 8400)"""
        detections = []
        
        # Get the output tensor
        if isinstance(outputs, dict):
            output_tensor = list(outputs.values())[0]
        else:
            output_tensor = outputs
        
        # YOLOv8 output shape is (1, 84, 8400) -> transpose to (8400, 84)
        # 84 = 4 (cx, cy, w, h) + 80 (class scores)
        predictions = output_tensor[0].T  # Shape: (8400, 84)
        
        # Unpack letterbox params
        r = self.ratio
        dw, dh = self.pad
        orig_h, orig_w = original_shape[:2]
        
        boxes = []
        confidences = []
        class_ids = []
        
        for pred in predictions:
            # First 4 values: cx, cy, w, h (in input image coordinates)
            cx, cy, w, h = pred[:4]
            # Remaining values: class scores
            class_scores = pred[4:]
            
            # Get best class
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]
            
            if confidence > self.confidence_threshold:
                # Convert from center format to corner format
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2
                
                # Undo letterbox: remove padding, then rescale by ratio
                x1 = (x1 - dw) / r
                y1 = (y1 - dh) / r
                x2 = (x2 - dw) / r
                y2 = (y2 - dh) / r
                
                # Clip to image boundaries
                x1 = max(0, min(int(x1), orig_w - 1))
                y1 = max(0, min(int(y1), orig_h - 1))
                x2 = max(0, min(int(x2), orig_w - 1))
                y2 = max(0, min(int(y2), orig_h - 1))
                
                box_w = x2 - x1
                box_h = y2 - y1
                
                boxes.append([x1, y1, box_w, box_h])
                confidences.append(float(confidence))
                class_ids.append(int(class_id))
        
        # Apply Non-Maximum Suppression
        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
            
            if len(indices) > 0:
                for i in indices.flatten():
                    x1, y1, w, h = boxes[i]
                    class_id = class_ids[i]
                    class_name = self.classes[class_id] if class_id < len(self.classes) else f"class_{class_id}"
                    
                    detections.append({
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": confidences[i],
                        "bbox_xyxy": [x1, y1, x1 + w, y1 + h]
                    })
        
        return detections
    
    def draw_detections(self, image, detections):
        """Draw bounding boxes and labels on image with unique colors per class"""
        annotated_image = image.copy()
        
        for detection in detections:
            class_id = detection.get('class_id', 0)
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # Get unique color for this class
            color = tuple(map(int, self.colors[class_id % len(self.colors)]))
            
            # Use the xyxy format from the new postprocessing
            if 'bbox_xyxy' in detection:
                x1, y1, x2, y2 = detection['bbox_xyxy']
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            else:
                # Fallback to old format
                bbox = detection['bbox']
                h, w = image.shape[:2]
                x_center, y_center, width, height = bbox
                x1 = int((x_center - width/2) * w)
                y1 = int((y_center - height/2) * h)
                x2 = int((x_center + width/2) * w)
                y2 = int((y_center + height/2) * h)
            
            # Ensure coordinates are within image bounds
            h, w = image.shape[:2]
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(0, min(x2, w-1))
            y2 = max(0, min(y2, h-1))
            
            # Draw bounding box with class-specific color
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background with class-specific color
            label = f"{class_name}: {confidence:.2f}"
            (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_image, (x1, y1 - label_h - 10), 
                         (x1 + label_w, y1), color, -1)
            cv2.putText(annotated_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return annotated_image
    
    def test_image(self, input_path, output_path):
        """Test YOLOv8n on a single image"""
        console.print(f"[blue]Testing YOLOv8n on image: {input_path}[/blue]")
        
        # Load image
        image = cv2.imread(input_path)
        if image is None:
            console.print(f"[red]❌ Could not load image: {input_path}[/red]")
            return False
        
        # Preprocess image
        input_data = self.preprocess_image(image)
        
        # Run inference
        start_time = time.time()
        input_tensor = self.model.input(0)
        output_tensor = self.model.output(0)
        
        result = self.model([input_data])
        outputs = result[output_tensor]
        
        inference_time = time.time() - start_time
        
        # Debug output
        # console.print(f"[yellow]Debug: Output shape: {outputs.shape}[/yellow]")
        # console.print(f"[yellow]Debug: Max output value: {np.max(outputs)}[/yellow]")
        # console.print(f"[yellow]Debug: Min output value: {np.min(outputs)}[/yellow]")
        # console.print(f"[yellow]Debug: Confidence threshold: {self.confidence_threshold}[/yellow]")
        
        # Debug: Check output structure
        # console.print(f"[yellow]Debug: Output dimensions: {outputs.shape}[/yellow]")
        # console.print(f"[yellow]Debug: Sample values from different positions:[/yellow]")
        # console.print(f"[yellow]  outputs[0, 0, 0] (x_center): {outputs[0, 0, 0]}[/yellow]")
        # console.print(f"[yellow]  outputs[0, 1, 0] (y_center): {outputs[0, 1, 0]}[/yellow]")
        # console.print(f"[yellow]  outputs[0, 2, 0] (width): {outputs[0, 2, 0]}[/yellow]")
        # console.print(f"[yellow]  outputs[0, 3, 0] (height): {outputs[0, 3, 0]}[/yellow]")
        # console.print(f"[yellow]  outputs[0, 4, 0] (objectness): {outputs[0, 4, 0]}[/yellow]")
        # console.print(f"[yellow]  outputs[0, 5, 0] (first class): {outputs[0, 5, 0]}[/yellow]")
        
        # Debug: Check if we have any high-confidence detections
        max_objectness = np.max(outputs[0, 4, :])
        max_class_score = np.max(outputs[0, 5:, :])
        # console.print(f"[yellow]Debug: Max objectness across all detections: {max_objectness:.4f}[/yellow]")
        # console.print(f"[yellow]Debug: Max class score across all detections: {max_class_score:.4f}[/yellow]")
        
        # Post-process detections
        detections = self.postprocess_detections(outputs, image.shape)
        
        # Draw detections
        annotated_image = self.draw_detections(image, detections)
        
        # Save result
        cv2.imwrite(output_path, annotated_image)
        
        # Print results
        console.print(f"[green]✅ Inference completed in {inference_time:.3f}s[/green]")
        console.print(f"[green]✅ Found {len(detections)} objects[/green]")
        console.print(f"[green]✅ Saved annotated image: {output_path}[/green]")
        
        # Print detection details
        for i, detection in enumerate(detections):
            console.print(f"[cyan]  {i+1}. {detection['class_name']}: {detection['confidence']:.3f}[/cyan]")
        
        return True
    
    def test_video(self, input_path, output_path, fps=None, inference_fps=None, length=None):
        """Test YOLOv8n on a video"""
        console.print(f"[blue]Testing YOLOv8 on video: {input_path}[/blue]")
        
        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            console.print(f"[red]❌ Could not open video: {input_path}[/red]")
            return False
        
        # Get video properties
        original_fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Determine output FPS:
        # 1. Use explicit --fps if provided
        # 2. Otherwise, use inference_fps to maintain correct playback speed
        # 3. Fall back to original fps if neither is specified
        if fps is not None:
            output_fps = fps
        elif inference_fps is not None:
            output_fps = inference_fps  # Match output fps to inference fps for correct playback
        else:
            output_fps = original_fps
        
        # Calculate frames to process based on length and inference FPS
        if length is not None:
            if inference_fps is not None:
                # Process only N frames per second for specified duration
                frames_to_process = int(length * inference_fps)
                total_duration = total_frames / original_fps
                console.print(f"[blue]Video properties: {width}x{height}, {original_fps} FPS (original), {output_fps} FPS (output)[/blue]")
                console.print(f"[blue]Processing {length}s at {inference_fps} FPS = {frames_to_process} frames out of {total_duration:.1f}s total[/blue]")
            else:
                # Process all frames for specified duration
                frames_to_process = int(length * original_fps)
                total_duration = total_frames / original_fps
                console.print(f"[blue]Video properties: {width}x{height}, {original_fps} FPS (original), {output_fps} FPS (output)[/blue]")
                console.print(f"[blue]Processing {length}s out of {total_duration:.1f}s total ({frames_to_process} frames)[/blue]")
        else:
            frames_to_process = total_frames
            console.print(f"[blue]Video properties: {width}x{height}, {original_fps} FPS (original), {output_fps} FPS (output), {total_frames} frames[/blue]")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
        
        frame_count = 0
        processed_frames = 0
        total_detections = 0
        
        # Calculate frame skip interval for inference FPS
        if inference_fps is not None:
            frame_skip = int(original_fps / inference_fps)
            console.print(f"[blue]Inference FPS: {inference_fps}, skipping {frame_skip-1} frames between inferences[/blue]")
        else:
            frame_skip = 1
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Processing video...", total=frames_to_process)
            
            while processed_frames < frames_to_process:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames based on inference FPS
                if frame_count % frame_skip == 0:
                    # Preprocess frame
                    input_data = self.preprocess_image(frame)
                    
                    # Run inference
                    input_tensor = self.model.input(0)
                    output_tensor = self.model.output(0)
                    
                    result = self.model([input_data])
                    outputs = result[output_tensor]
                    
                    # Post-process detections
                    detections = self.postprocess_detections(outputs, frame.shape)
                    
                    # Draw detections
                    annotated_frame = self.draw_detections(frame, detections)
                    
                    # Write frame
                    out.write(annotated_frame)
                    
                    processed_frames += 1
                    total_detections += len(detections)
                    progress.update(task, advance=1)
                
                frame_count += 1
        
        cap.release()
        out.release()
        
        console.print(f"[green]✅ Processed {processed_frames} frames (out of {frame_count} total frames)[/green]")
        console.print(f"[green]✅ Total detections: {total_detections}[/green]")
        console.print(f"[green]✅ Average detections per frame: {total_detections/processed_frames:.2f}[/green]")
        console.print(f"[green]✅ Saved annotated video: {output_path}[/green]")
        
        return True

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Object Detection Test")
    parser.add_argument("--input", type=str, required=True, help="Input image or video path")
    parser.add_argument("--output", type=str, help="Output path")
    parser.add_argument("--video", action="store_true", help="Process as video")
    parser.add_argument("--fps", type=int, help="Output video FPS (overrides original video FPS)")
    parser.add_argument("--inference-fps", type=float, help="Inference FPS - process only N frames per second (default: process all frames)")
    parser.add_argument("--length", type=float, help="Number of seconds to process from video (default: process entire video)")
    parser.add_argument("--size", choices=['n', 's', 'm'], default='n', help="YOLOv8 model size: n (nano), s (small), m (medium)")
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = YOLOv8Tester(model_size=args.size)
    
    # Load model
    if not tester.load_model():
        return
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        if args.video or args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            output_path = f"output/yolov8{args.size}_video_output.mp4"
        else:
            output_path = f"output/yolov8{args.size}_image_output.jpg"
    
    # Run test
    if args.video or args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        success = tester.test_video(args.input, output_path, args.fps, args.inference_fps, args.length)
    else:
        success = tester.test_image(args.input, output_path)
    
    if success:
        console.print(f"[green]🎉 YOLOv8{args.size.upper()} test completed successfully![/green]")
    else:
        console.print(f"[red]❌ YOLOv8{args.size.upper()} test failed![/red]")

if __name__ == "__main__":
    main()
