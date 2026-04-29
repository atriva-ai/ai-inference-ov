#!/usr/bin/env python3
"""
LPRNet License Plate Recognition Test
====================================

This script tests the LPRNet license plate recognition model with OpenVINO.
It can process both images and videos, recognizing license plates and creating annotated outputs.

Usage:
    python test_lprnet.py --input test_images/sample_license_plate.jpg
    python test_lprnet.py --input test_videos/sample_parking.mp4 --video
"""

import os
import sys
import json
import argparse
import cv2
import numpy as np
from pathlib import Path
from openvino import Core
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
import time

console = Console()

class LPRNetTester:
    """LPRNet License Plate Recognition Tester"""
    
    def __init__(self):
        self.models_dir = Path("../models/lprnet")
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Load model configuration
        with open(self.models_dir / "model.json", 'r') as f:
            self.config = json.load(f)
        
        # Initialize OpenVINO
        self.ie = Core()
        self.model = None
        self.input_shape = self.config["input_shape"][2:]  # [24, 94]
        self.classes = self.config["classes"]
        self.confidence_threshold = self.config["confidence_threshold"]
        self.max_plate_length = self.config["max_plate_length"]
        
        console.print(f"[green]LPRNet Tester initialized[/green]")
        console.print(f"[blue]Input shape: {self.input_shape}[/blue]")
        console.print(f"[blue]Character classes: {len(self.classes)}[/blue]")
        console.print(f"[blue]Max plate length: {self.max_plate_length}[/blue]")
    
    def load_model(self):
        """Load LPRNet OpenVINO model"""
        xml_path = self.models_dir / "lprnet.xml"
        bin_path = self.models_dir / "lprnet.bin"
        
        if not xml_path.exists() or not bin_path.exists():
            console.print("[red]Model files not found. Please download the model first.[/red]")
            return False
        
        try:
            model = self.ie.read_model(model=xml_path)
            self.model = self.ie.compile_model(model=model, device_name="CPU")
            console.print("[green]✅ LPRNet model loaded successfully[/green]")
            return True
        except Exception as e:
            console.print(f"[red]❌ Failed to load model: {str(e)}[/red]")
            return False
    
    def preprocess_image(self, image):
        """Preprocess image for LPRNet inference"""
        # Resize image to model input size
        resized_image = cv2.resize(image, self.input_shape)
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized_image = rgb_image.astype(np.float32) / 255.0
        
        # Transpose to CHW format
        input_data = np.transpose(normalized_image, (2, 0, 1))
        input_data = np.expand_dims(input_data, axis=0)
        
        return input_data
    
    def postprocess_recognition(self, outputs):
        """Post-process LPRNet recognition results"""
        # This is a simplified post-processing
        # Real implementation would include proper CTC decoding
        
        output_shape = outputs.shape  # [1, 18, 37]
        batch_size, seq_len, num_classes = output_shape
        
        # Decode sequence (simplified)
        decoded_text = ""
        for i in range(seq_len):
            class_probs = outputs[0, i, :]
            class_id = np.argmax(class_probs)
            confidence = class_probs[class_id]
            
            if confidence > self.confidence_threshold and class_id < len(self.classes):
                char = self.classes[class_id]
                if char != ' ':  # Skip spaces
                    decoded_text += char
        
        return decoded_text
    
    def detect_license_plates(self, image):
        """Detect license plate regions in image (simplified)"""
        # This is a simplified license plate detection
        # Real implementation would use a separate detection model
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours that could be license plates
        plate_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            # License plates typically have aspect ratio between 2.0 and 5.0
            if 2.0 <= aspect_ratio <= 5.0 and w > 100 and h > 20:
                plate_regions.append((x, y, w, h))
        
        return plate_regions
    
    def draw_recognition(self, image, plate_regions, recognized_texts):
        """Draw license plate recognition results on image"""
        annotated_image = image.copy()
        
        for i, (x, y, w, h) in enumerate(plate_regions):
            # Draw bounding box
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw recognized text
            if i < len(recognized_texts):
                text = recognized_texts[i]
                if text:
                    # Draw text background
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(annotated_image, (x, y - text_size[1] - 10), 
                                 (x + text_size[0], y), (0, 255, 0), -1)
                    cv2.putText(annotated_image, text, (x, y - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        return annotated_image
    
    def test_image(self, input_path, output_path):
        """Test LPRNet on a single image"""
        console.print(f"[blue]Testing LPRNet on image: {input_path}[/blue]")
        
        # Load image
        image = cv2.imread(input_path)
        if image is None:
            console.print(f"[red]❌ Could not load image: {input_path}[/red]")
            return False
        
        # Detect license plate regions
        plate_regions = self.detect_license_plates(image)
        console.print(f"[blue]Found {len(plate_regions)} potential license plate regions[/blue]")
        
        recognized_texts = []
        
        # Process each detected region
        for i, (x, y, w, h) in enumerate(plate_regions):
            # Extract license plate region
            plate_region = image[y:y+h, x:x+w]
            
            # Preprocess for LPRNet
            input_data = self.preprocess_image(plate_region)
            
            # Run inference
            start_time = time.time()
            input_tensor = self.model.input(0)
            output_tensor = self.model.output(0)
            
            result = self.model([input_tensor], {input_tensor: input_data})
            outputs = result[output_tensor]
            
            inference_time = time.time() - start_time
            
            # Post-process recognition
            recognized_text = self.postprocess_recognition(outputs)
            recognized_texts.append(recognized_text)
            
            console.print(f"[cyan]Region {i+1}: '{recognized_text}' (inference: {inference_time:.3f}s)[/cyan]")
        
        # Draw results
        annotated_image = self.draw_recognition(image, plate_regions, recognized_texts)
        
        # Save result
        cv2.imwrite(output_path, annotated_image)
        
        console.print(f"[green]✅ Recognition completed[/green]")
        console.print(f"[green]✅ Found {len(plate_regions)} license plates[/green]")
        console.print(f"[green]✅ Saved annotated image: {output_path}[/green]")
        
        return True
    
    def test_video(self, input_path, output_path):
        """Test LPRNet on a video"""
        console.print(f"[blue]Testing LPRNet on video: {input_path}[/blue]")
        
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
        total_plates = 0
        
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
                
                # Detect license plate regions
                plate_regions = self.detect_license_plates(frame)
                
                recognized_texts = []
                
                # Process each detected region
                for x, y, w, h in plate_regions:
                    # Extract license plate region
                    plate_region = frame[y:y+h, x:x+w]
                    
                    # Preprocess for LPRNet
                    input_data = self.preprocess_image(plate_region)
                    
                    # Run inference
                    input_tensor = self.model.input(0)
                    output_tensor = self.model.output(0)
                    
                    result = self.model([input_tensor], {input_tensor: input_data})
                    outputs = result[output_tensor]
                    
                    # Post-process recognition
                    recognized_text = self.postprocess_recognition(outputs)
                    recognized_texts.append(recognized_text)
                
                # Draw results
                annotated_frame = self.draw_recognition(frame, plate_regions, recognized_texts)
                
                # Write frame
                out.write(annotated_frame)
                
                frame_count += 1
                total_plates += len(plate_regions)
                progress.update(task, advance=1)
        
        cap.release()
        out.release()
        
        console.print(f"[green]✅ Processed {frame_count} frames[/green]")
        console.print(f"[green]✅ Total license plates detected: {total_plates}[/green]")
        console.print(f"[green]✅ Average plates per frame: {total_plates/frame_count:.2f}[/green]")
        console.print(f"[green]✅ Saved annotated video: {output_path}[/green]")
        
        return True

def main():
    parser = argparse.ArgumentParser(description="LPRNet License Plate Recognition Test")
    parser.add_argument("--input", type=str, required=True, help="Input image or video path")
    parser.add_argument("--output", type=str, help="Output path")
    parser.add_argument("--video", action="store_true", help="Process as video")
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = LPRNetTester()
    
    # Load model
    if not tester.load_model():
        return
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        if args.video or args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            output_path = "output/lprnet_video_output.mp4"
        else:
            output_path = "output/lprnet_image_output.jpg"
    
    # Run test
    if args.video or args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        success = tester.test_video(args.input, output_path)
    else:
        success = tester.test_image(args.input, output_path)
    
    if success:
        console.print("[green]🎉 LPRNet test completed successfully![/green]")
    else:
        console.print("[red]❌ LPRNet test failed![/red]")

if __name__ == "__main__":
    main()
