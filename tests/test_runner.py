#!/usr/bin/env python3
"""
OpenVINO AI Model Testing Suite
===============================

This script provides comprehensive testing for OpenVINO AI models including:
- YOLOv8n Object Detection
- LPRNet License Plate Recognition  
- Vehicle Detection and Tracking

Usage:
    python test_runner.py --model yolov8n --input test_images/sample.jpg
    python test_runner.py --model all --input test_videos/sample.mp4
    python test_runner.py --download-models
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import cv2
import numpy as np
from openvino import Core
import requests
import yaml
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

class OpenVINOModelTester:
    """Main testing class for OpenVINO AI models"""
    
    def __init__(self, models_dir: str = "../models"):
        self.models_dir = Path(models_dir)
        self.test_images_dir = Path("test_images")
        self.test_videos_dir = Path("test_videos")
        self.output_dir = Path("output")
        
        # Create directories if they don't exist
        self._create_directories()
        
        # Initialize OpenVINO Core
        self.ie = Core()
        self.available_devices = self.ie.available_devices
        console.print(f"[green]Available OpenVINO devices: {self.available_devices}[/green]")
        
    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            self.models_dir,
            self.test_images_dir,
            self.test_videos_dir,
            self.output_dir
        ]
        
        for directory in directories:
            directory.mkdir(exist_ok=True)
            console.print(f"[blue]Created directory: {directory}[/blue]")
    
    def load_model_config(self, model_name: str) -> Dict:
        """Load model configuration from JSON file"""
        config_path = self.models_dir / model_name / "model.json"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Model config not found: {config_path}")
        
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def download_model(self, model_name: str) -> bool:
        """Download and convert model to OpenVINO format"""
        config = self.load_model_config(model_name)
        model_dir = self.models_dir / model_name
        
        console.print(f"[yellow]Downloading {model_name} model...[/yellow]")
        
        try:
            # For YOLOv8 models, use ultralytics to download and convert
            if model_name.startswith("yolov8"):
                return self._download_and_convert_yolov8(model_name, model_dir)
            
            # For other models, try direct download from OpenVINO model zoo
            base_url = config.get("openvino_model_url", "").rstrip("/")
            if not base_url:
                console.print(f"[red]❌ No download URL configured for {model_name}[/red]")
                return False
                
            xml_url = f"{base_url}/{model_name}.xml"
            xml_path = model_dir / f"{model_name}.xml"
            
            response = requests.get(xml_url, stream=True)
            response.raise_for_status()
            
            with open(xml_path, 'wb') as f:
                for chunk in tqdm(response.iter_content(chunk_size=8192), desc="Downloading XML"):
                    f.write(chunk)
            
            # Download BIN file
            bin_url = f"{base_url}/{model_name}.bin"
            bin_path = model_dir / f"{model_name}.bin"
            
            response = requests.get(bin_url, stream=True)
            response.raise_for_status()
            
            with open(bin_path, 'wb') as f:
                for chunk in tqdm(response.iter_content(chunk_size=8192), desc="Downloading BIN"):
                    f.write(chunk)
            
            console.print(f"[green]✅ Successfully downloaded {model_name} model[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Failed to download {model_name}: {str(e)}[/red]")
            return False
    
    def _download_and_convert_yolov8(self, model_name: str, model_dir: Path) -> bool:
        """Download YOLOv8 model and convert to OpenVINO format using ultralytics"""
        try:
            from ultralytics import YOLO
        except ImportError:
            console.print("[red]❌ ultralytics package not installed. Run: pip install ultralytics[/red]")
            return False
        
        try:
            console.print(f"[yellow]Downloading {model_name} from Ultralytics...[/yellow]")
            
            # Load the model (this downloads the .pt file automatically)
            model = YOLO(f"{model_name}.pt")
            
            # Export to OpenVINO format
            console.print(f"[yellow]Converting {model_name} to OpenVINO format...[/yellow]")
            export_path = model.export(format="openvino")
            
            # Move the exported files to our model directory
            export_dir = Path(export_path)
            xml_src = export_dir / f"{model_name}.xml"
            bin_src = export_dir / f"{model_name}.bin"
            
            xml_dst = model_dir / f"{model_name}.xml"
            bin_dst = model_dir / f"{model_name}.bin"
            
            # Copy files to model directory
            import shutil
            if xml_src.exists():
                shutil.copy2(xml_src, xml_dst)
            if bin_src.exists():
                shutil.copy2(bin_src, bin_dst)
            
            console.print(f"[green]✅ Successfully downloaded and converted {model_name} to OpenVINO format[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Failed to download/convert {model_name}: {str(e)}[/red]")
            return False
    
    def _load_yolov8_classes(self, model_name: str) -> List[str]:
        """Load class names from ultralytics metadata.yaml file"""
        model_dir = self.models_dir / model_name
        
        # Search for metadata.yaml - prioritize ultralytics subdirectory (COCO model)
        search_paths = [
            model_dir / "ultralytics" / "metadata.yaml",
            model_dir / f"{model_name}_openvino_model" / "metadata.yaml",
            model_dir / "metadata.yaml",  # Fallback to root (may be custom-trained)
        ]
        
        for metadata_path in search_paths:
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = yaml.safe_load(f)
                    
                    # Extract class names from 'names' dict
                    names_dict = metadata.get('names', {})
                    if names_dict:
                        # Convert dict {0: 'person', 1: 'bicycle', ...} to list
                        max_idx = max(names_dict.keys())
                        classes = [''] * (max_idx + 1)
                        for idx, name in names_dict.items():
                            classes[idx] = name
                        console.print(f"[blue]Loaded {len(classes)} class names from {metadata_path}[/blue]")
                        return classes
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not load metadata from {metadata_path}: {e}[/yellow]")
        
        return []
    
    def load_openvino_model(self, model_name: str):
        """Load OpenVINO model for inference"""
        config = self.load_model_config(model_name)
        model_dir = self.models_dir / model_name
        
        xml_path = model_dir / f"{model_name}.xml"
        bin_path = model_dir / f"{model_name}.bin"
        
        if not xml_path.exists() or not bin_path.exists():
            console.print(f"[red]Model files not found for {model_name}. Downloading...[/red]")
            if not self.download_model(model_name):
                raise FileNotFoundError(f"Could not download {model_name} model")
        
        # Load model
        model = self.ie.read_model(model=xml_path)
        compiled_model = self.ie.compile_model(model=model, device_name="CPU")
        
        console.print(f"[green]✅ Loaded {model_name} model successfully[/green]")
        return compiled_model, config
    
    def test_yolov8n_detection(self, input_path: str, output_path: str) -> bool:
        """Test YOLOv8n object detection"""
        console.print("[blue]Testing YOLOv8n Object Detection...[/blue]")
        
        try:
            model, config = self.load_openvino_model("yolov8n")
            
            # Load and preprocess image
            image = cv2.imread(input_path)
            if image is None:
                raise ValueError(f"Could not load image: {input_path}")
            
            orig_h, orig_w = image.shape[:2]
            
            # Resize image to model input size
            input_shape = config["input_shape"][2:]  # [640, 640]
            input_h, input_w = input_shape
            resized_image = cv2.resize(image, (input_w, input_h))
            
            # Convert BGR to RGB and normalize
            rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            normalized_image = rgb_image.astype(np.float32) / 255.0
            
            # Transpose to CHW format
            input_data = np.transpose(normalized_image, (2, 0, 1))
            input_data = np.expand_dims(input_data, axis=0)
            
            # Run inference
            output_tensor = model.output(0)
            
            result = model(input_data)
            outputs = result[output_tensor]
            
            console.print(f"[green]✅ YOLOv8n inference completed. Output shape: {outputs.shape}[/green]")
            
            # Post-process YOLOv8 outputs
            # Output shape is (1, 84, 8400) -> transpose to (8400, 84)
            predictions = outputs[0].T  # Shape: (8400, 84)
            
            conf_threshold = config.get("confidence_threshold", 0.25)
            nms_threshold = config.get("nms_threshold", 0.45)
            
            # Load class names from metadata.yaml (ultralytics export) if available
            classes = self._load_yolov8_classes("yolov8n")
            if not classes:
                classes = config.get("classes", [])
            
            boxes = []
            confidences = []
            class_ids = []
            
            # Scale factors
            scale_x = orig_w / input_w
            scale_y = orig_h / input_h
            
            for pred in predictions:
                # First 4 values: cx, cy, w, h
                cx, cy, w, h = pred[:4]
                # Remaining 80 values: class scores
                class_scores = pred[4:]
                
                # Get best class
                class_id = np.argmax(class_scores)
                confidence = class_scores[class_id]
                
                if confidence > conf_threshold:
                    # Convert from center format to corner format and scale to original image
                    x1 = int((cx - w / 2) * scale_x)
                    y1 = int((cy - h / 2) * scale_y)
                    box_w = int(w * scale_x)
                    box_h = int(h * scale_y)
                    
                    boxes.append([x1, y1, box_w, box_h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
            
            # Apply Non-Maximum Suppression
            indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
            
            # Draw bounding boxes
            annotated_image = image.copy()
            
            # Generate colors for each class
            np.random.seed(42)
            colors = np.random.randint(0, 255, size=(len(classes), 3), dtype=np.uint8)
            
            detection_count = 0
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    class_id = class_ids[i]
                    confidence = confidences[i]
                    
                    # Get class name and color
                    class_name = classes[class_id] if class_id < len(classes) else f"class_{class_id}"
                    color = tuple(map(int, colors[class_id % len(colors)]))
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_image, (x, y), (x + w, y + h), color, 2)
                    
                    # Draw label background
                    label = f"{class_name}: {confidence:.2f}"
                    (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(annotated_image, (x, y - label_h - 10), (x + label_w, y), color, -1)
                    
                    # Draw label text
                    cv2.putText(annotated_image, label, (x, y - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    detection_count += 1
            
            console.print(f"[green]✅ Detected {detection_count} objects[/green]")
            
            cv2.imwrite(output_path, annotated_image)
            console.print(f"[green]✅ Saved annotated image: {output_path}[/green]")
            
            return True
            
        except Exception as e:
            console.print(f"[red]❌ YOLOv8n test failed: {str(e)}[/red]")
            return False
    
    def _recognize_license_plate(self, crop_image) -> str:
        """Run text recognition on a cropped license plate using EasyOCR (best accuracy for Western plates)"""
        try:
            import easyocr
            
            # Initialize EasyOCR reader (lazy loading)
            if not hasattr(self, '_ocr_reader'):
                console.print("[cyan]   Loading EasyOCR...[/cyan]")
                self._ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            
            # Run OCR on the crop
            results = self._ocr_reader.readtext(crop_image, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
            
            if results:
                # Get all detected text
                texts = []
                for (bbox, text, conf) in results:
                    clean_text = ''.join(c for c in text.upper() if c.isalnum())
                    if len(clean_text) >= 4:
                        texts.append((clean_text, conf))
                
                if texts:
                    best_text, best_conf = max(texts, key=lambda x: x[1])
                    return f"{best_text} ({best_conf:.0%})"
            
            return "(no text detected)"
                
        except ImportError:
            # Fallback to OpenVINO text-recognition-0012
            console.print("[yellow]   EasyOCR not available, using OpenVINO text-recognition-0012...[/yellow]")
            return self._recognize_plate_openvino(crop_image)
        except Exception as e:
            console.print(f"[yellow]   ⚠️ EasyOCR failed: {str(e)}, trying OpenVINO...[/yellow]")
            return self._recognize_plate_openvino(crop_image)
    
    def _recognize_plate_openvino(self, crop_image) -> str:
        """Fallback recognition using OpenVINO text-recognition-0012"""
        try:
            if not hasattr(self, '_text_rec_model'):
                self._text_rec_model, _ = self.load_openvino_model("text-recognition-0012")
            
            # Convert to grayscale
            if len(crop_image.shape) == 3:
                gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = crop_image
            
            # Resize to 120x32 (model expects width=120, height=32)
            resized = cv2.resize(gray, (120, 32))
            
            # Normalize to [0, 1] and reshape
            normalized = resized.astype(np.float32) / 255.0
            input_data = normalized.reshape(1, 32, 120, 1)
            
            # Run inference
            result = self._text_rec_model(input_data)
            outputs = result[self._text_rec_model.output(0)]
            
            # Decode CTC output [30, 1, 37]
            chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ#"
            predictions = outputs[:, 0, :]
            
            # Apply softmax for proper probabilities
            exp_preds = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
            probs = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)
            
            indices = np.argmax(probs, axis=1)
            
            # CTC decode
            plate_text = ""
            prev_idx = -1
            conf_sum = 0
            conf_count = 0
            for i, idx in enumerate(indices):
                if idx != 36 and idx != prev_idx:  # 36 is blank
                    if idx < len(chars) - 1:
                        plate_text += chars[idx]
                        conf_sum += probs[i, idx]
                        conf_count += 1
                prev_idx = idx
            
            if plate_text and conf_count > 0:
                avg_conf = conf_sum / conf_count
                return f"{plate_text} ({avg_conf:.0%}) [OpenVINO]"
            
            return "(no text detected)"
        except Exception as e:
            return f"(error: {str(e)})"
    
    def test_ssd_detection(self, model_name: str, input_path: str, output_path: str, classes: list) -> bool:
        """Test SSD-style detection models (face detection, license plate detection)"""
        console.print(f"[blue]Testing {model_name}...[/blue]")
        
        try:
            model, config = self.load_openvino_model(model_name)
            
            # Load and preprocess image
            image = cv2.imread(input_path)
            if image is None:
                raise ValueError(f"Could not load image: {input_path}")
            
            orig_h, orig_w = image.shape[:2]
            console.print(f"[cyan]Original image size: {orig_w}x{orig_h}[/cyan]")
            
            # Get model input shape and determine format
            model_input_shape = model.input(0).shape
            console.print(f"[cyan]Model input shape: {model_input_shape}[/cyan]")
            
            # Determine if model expects NCHW or NHWC
            if len(model_input_shape) == 4:
                if model_input_shape[1] == 3:  # NCHW format
                    input_h, input_w = model_input_shape[2], model_input_shape[3]
                    is_nchw = True
                else:  # NHWC format
                    input_h, input_w = model_input_shape[1], model_input_shape[2]
                    is_nchw = False
            else:
                input_h, input_w = 300, 300
                is_nchw = True
            
            resized_image = cv2.resize(image, (input_w, input_h))
            
            # OpenVINO SSD models expect BGR format in [0, 255] range (NOT normalized RGB)
            # Keep as BGR (OpenCV loads as BGR by default)
            input_image = resized_image.astype(np.float32)
            
            # Format input data based on model expectation
            if is_nchw:
                input_data = np.transpose(input_image, (2, 0, 1))
                input_data = np.expand_dims(input_data, axis=0)
            else:
                input_data = np.expand_dims(input_image, axis=0)
            
            console.print(f"[cyan]Input data shape: {input_data.shape}, range: [{input_data.min():.1f}, {input_data.max():.1f}][/cyan]")
            
            # Run inference
            output_tensor = model.output(0)
            result = model(input_data)
            outputs = result[output_tensor]
            
            console.print(f"[green]✅ {model_name} inference completed. Output shape: {outputs.shape}[/green]")
            
            # Post-process SSD outputs
            # Output shape is (1, 1, N, 7) where each detection is:
            # [image_id, label, confidence, x_min, y_min, x_max, y_max]
            conf_threshold = config.get("confidence_threshold", 0.5)
            
            # Debug: Show top detections by confidence
            all_confs = [(outputs[0, 0, i, 2], outputs[0, 0, i, 1]) for i in range(outputs.shape[2])]
            top_confs = sorted(all_confs, reverse=True)[:10]
            console.print(f"[yellow]Top 10 raw confidences: {[(f'{c:.3f}', int(l)) for c, l in top_confs]}[/yellow]")
            console.print(f"[yellow]Using confidence threshold: {conf_threshold}[/yellow]")
            
            detections = []
            for detection in outputs[0, 0]:
                _, label, conf, x_min, y_min, x_max, y_max = detection
                
                if conf > conf_threshold and int(label) > 0:  # Skip background (label 0)
                    # Convert normalized coords to pixel coords
                    x1 = int(x_min * orig_w)
                    y1 = int(y_min * orig_h)
                    x2 = int(x_max * orig_w)
                    y2 = int(y_max * orig_h)
                    
                    # Clip to image boundaries
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(orig_w, x2), min(orig_h, y2)
                    
                    class_id = int(label)
                    class_name = classes[class_id] if class_id < len(classes) else f"class_{class_id}"
                    
                    detections.append({
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": float(conf),
                        "bbox": [x1, y1, x2, y2]
                    })
            
            console.print(f"[green]Found {len(detections)} detections[/green]")
            
            # Draw detections on image
            output_image = image.copy()
            colors = {
                "face": (0, 255, 0),
                "vehicle": (255, 0, 0),
                "license_plate": (0, 255, 255),
            }
            
            crop_idx = 0
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                class_name = det["class_name"]
                conf = det["confidence"]
                
                color = colors.get(class_name, (0, 255, 0))
                cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
                
                label_text = f"{class_name}: {conf:.2f}"
                (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(output_image, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
                cv2.putText(output_image, label_text, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                console.print(f"  {class_name}: {conf:.2f} @ [{x1}, {y1}, {x2}, {y2}]")
                
                # Save cropped license plates and run recognition
                if class_name == "license_plate":
                    crop = image[y1:y2, x1:x2]
                    if crop.size > 0:
                        crop_path = str(self.output_dir / f"license_plate_crop_{crop_idx}.jpg")
                        cv2.imwrite(crop_path, crop)
                        console.print(f"[cyan]   📷 Saved crop to: {crop_path} (size: {x2-x1}x{y2-y1})[/cyan]")
                        
                        # Run LPRNet recognition on the crop
                        plate_text = self._recognize_license_plate(crop)
                        if plate_text:
                            console.print(f"[green]   🔤 Recognized plate: {plate_text}[/green]")
                            # Draw plate text on output image
                            cv2.putText(output_image, plate_text, (x1, y2 + 25), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        
                        crop_idx += 1
            
            # Save output
            cv2.imwrite(output_path, output_image)
            console.print(f"[green]✅ Saved output to: {output_path}[/green]")
            
            return True
            
        except Exception as e:
            console.print(f"[red]❌ {model_name} test failed: {str(e)}[/red]")
            import traceback
            traceback.print_exc()
            return False
    
    def test_lprnet_recognition(self, input_path: str, output_path: str) -> bool:
        """Test LPRNet license plate recognition"""
        console.print("[blue]Testing LPRNet License Plate Recognition...[/blue]")
        
        try:
            model, config = self.load_openvino_model("lprnet")
            
            # Load and preprocess image
            image = cv2.imread(input_path)
            if image is None:
                raise ValueError(f"Could not load image: {input_path}")
            
            orig_h, orig_w = image.shape[:2]
            console.print(f"[cyan]Input image size: {orig_w}x{orig_h}[/cyan]")
            
            # Get model input shape: [1, 3, 24, 94] -> resize to (94, 24) (width, height)
            model_input_shape = model.input(0).shape
            console.print(f"[cyan]Model input shape: {model_input_shape}[/cyan]")
            
            input_h, input_w = model_input_shape[2], model_input_shape[3]  # 24, 94
            resized_image = cv2.resize(image, (input_w, input_h))
            
            # Normalize to [0, 1] and convert to NCHW
            normalized_image = resized_image.astype(np.float32) / 255.0
            input_data = np.transpose(normalized_image, (2, 0, 1))
            input_data = np.expand_dims(input_data, axis=0)
            
            console.print(f"[cyan]Input data shape: {input_data.shape}[/cyan]")
            
            # This model requires 2 inputs: data and seq_ind
            # seq_ind is sequence indices for decoding
            seq_ind = np.arange(88).reshape(88, 1).astype(np.float32)
            
            # Run inference with both inputs
            output_tensor = model.output(0)
            result = model({"data": input_data, "seq_ind": seq_ind})
            outputs = result[output_tensor]
            
            console.print(f"[green]✅ LPRNet inference completed. Output shape: {outputs.shape}[/green]")
            
            # Decode output - shape is [1, 88, 1, 1]
            # 88 classes: 0-9, Chinese provinces, A-Z, <blank>
            logits = outputs.squeeze()  # Shape: (88,)
            
            # Character set for license-plate-recognition-barrier-0001
            chars = "0123456789" + \
                    "<Anhui><Beijing><Chongqing><Fujian><Gansu><Guangdong>" + \
                    "<Guangxi><Guizhou><Hainan><Hebei><Heilongjiang><Henan>" + \
                    "<HongKong><Hubei><Hunan><InnerMongolia><Jiangsu><Jiangxi>" + \
                    "<Jilin><Liaoning><Macau><Ningxia><Qinghai><Shaanxi>" + \
                    "<Shandong><Shanghai><Shanxi><Sichuan><Tianjin><Tibet>" + \
                    "<Xinjiang><Yunnan><Zhejiang><police>" + \
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            
            # Simple decoding - take top prediction
            top_idx = np.argmax(logits)
            top_conf = logits[top_idx]
            
            console.print(f"[yellow]Top 5 predictions:[/yellow]")
            top5_indices = np.argsort(logits)[::-1][:5]
            for idx in top5_indices:
                if idx < 10:
                    char = str(idx)
                elif idx < 44:
                    char = f"<Province_{idx-10}>"
                elif idx < 70:
                    char = chr(ord('A') + idx - 44)
                else:
                    char = "<blank>"
                console.print(f"  [{idx}] {char}: {logits[idx]:.4f}")
            
            # Note: This model outputs single character - full plate recognition needs sequence model
            console.print(f"[yellow]Note: This model is designed for Chinese license plates[/yellow]")
            console.print(f"[yellow]      Output represents confidence per character class[/yellow]")
            
            # Save annotated image
            annotated_image = image.copy()
            # Scale up for better visibility
            annotated_image = cv2.resize(annotated_image, (orig_w * 2, orig_h * 2))
            cv2.putText(annotated_image, f"LPRNet Recognition", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imwrite(output_path, annotated_image)
            console.print(f"[green]✅ Saved annotated image: {output_path}[/green]")
            
            return True
            
        except Exception as e:
            console.print(f"[red]❌ LPRNet test failed: {str(e)}[/red]")
            import traceback
            traceback.print_exc()
            return False
    
    def test_vehicle_tracking(self, input_path: str, output_path: str) -> bool:
        """Test vehicle detection and tracking"""
        console.print("[blue]Testing Vehicle Detection and Tracking...[/blue]")
        
        try:
            model, config = self.load_openvino_model("vehicle_tracking")
            
            # Load and preprocess image
            image = cv2.imread(input_path)
            if image is None:
                raise ValueError(f"Could not load image: {input_path}")
            
            # Resize image to model input size
            input_shape = config["input_shape"][2:]  # [416, 416]
            resized_image = cv2.resize(image, input_shape)
            
            # Convert BGR to RGB and normalize
            rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            normalized_image = rgb_image.astype(np.float32) / 255.0
            
            # Transpose to CHW format
            input_data = np.transpose(normalized_image, (2, 0, 1))
            input_data = np.expand_dims(input_data, axis=0)
            
            # Run inference
            output_tensor = model.output(0)
            
            result = model(input_data)
            outputs = result[output_tensor]
            
            # Process outputs (simplified - would need proper post-processing)
            console.print(f"[green]✅ Vehicle tracking inference completed. Output shape: {outputs.shape}[/green]")
            
            # Save annotated image
            annotated_image = image.copy()
            cv2.putText(annotated_image, "Vehicle Tracking", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imwrite(output_path, annotated_image)
            console.print(f"[green]✅ Saved annotated image: {output_path}[/green]")
            
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Vehicle tracking test failed: {str(e)}[/red]")
            return False
    
    def test_video_processing(self, model_name: str, input_path: str, output_path: str) -> bool:
        """Test video processing with specified model"""
        console.print(f"[blue]Testing {model_name} on video: {input_path}[/blue]")
        
        try:
            # Only YOLOv8n is fully implemented for video
            if model_name != "yolov8n":
                console.print(f"[yellow]⚠️ Video processing only implemented for yolov8n, skipping {model_name}[/yellow]")
                return False
            
            # Load model and config
            model, config = self.load_openvino_model(model_name)
            classes = self._load_yolov8_classes(model_name)
            if not classes:
                classes = config.get("classes", [])
            
            conf_threshold = config.get("confidence_threshold", 0.25)
            nms_threshold = config.get("nms_threshold", 0.45)
            input_shape = config["input_shape"][2:]  # [640, 640]
            input_h, input_w = input_shape
            
            # Generate colors for each class
            np.random.seed(42)
            colors = np.random.randint(0, 255, size=(len(classes), 3), dtype=np.uint8)
            
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {input_path}")
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            console.print(f"[blue]Video: {width}x{height} @ {fps}fps, {total_frames} frames[/blue]")
            
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            output_tensor = model.output(0)
            scale_x = width / input_w
            scale_y = height / input_h
            
            frame_count = 0
            for _ in tqdm(range(total_frames), desc="Processing frames"):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Preprocess frame
                resized = cv2.resize(frame, (input_w, input_h))
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                normalized = rgb.astype(np.float32) / 255.0
                input_data = np.transpose(normalized, (2, 0, 1))
                input_data = np.expand_dims(input_data, axis=0)
                
                # Run inference
                result = model(input_data)
                outputs = result[output_tensor]
                predictions = outputs[0].T
                
                # Post-process detections
                boxes, confidences, class_ids = [], [], []
                for pred in predictions:
                    cx, cy, w, h = pred[:4]
                    class_scores = pred[4:]
                    class_id = np.argmax(class_scores)
                    confidence = class_scores[class_id]
                    
                    if confidence > conf_threshold:
                        x1 = int((cx - w / 2) * scale_x)
                        y1 = int((cy - h / 2) * scale_y)
                        box_w = int(w * scale_x)
                        box_h = int(h * scale_y)
                        boxes.append([x1, y1, box_w, box_h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
                
                # Apply NMS and draw boxes
                indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
                
                if len(indices) > 0:
                    for i in indices.flatten():
                        x, y, w, h = boxes[i]
                        class_id = class_ids[i]
                        confidence = confidences[i]
                        class_name = classes[class_id] if class_id < len(classes) else f"class_{class_id}"
                        color = tuple(map(int, colors[class_id % len(colors)]))
                        
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        label = f"{class_name}: {confidence:.2f}"
                        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(frame, (x, y - lh - 10), (x + lw, y), color, -1)
                        cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                out.write(frame)
                frame_count += 1
            
            cap.release()
            out.release()
            
            console.print(f"[green]✅ Processed video: {frame_count} frames saved to {output_path}[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Video processing failed: {str(e)}[/red]")
            return False
    
    def run_all_tests(self, input_path: str) -> Dict[str, bool]:
        """Run all model tests"""
        results = {}
        
        # Determine if input is image or video
        is_video = input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
        
        if is_video:
            # Test video processing (only YOLOv8n for now, others need specific handling)
            output_path = self.output_dir / "yolov8n_video_output.mp4"
            results["yolov8n"] = self.test_video_processing("yolov8n", input_path, str(output_path))
        else:
            # Test image processing - all detection models
            results["yolov8n"] = self.test_yolov8n_detection(input_path, 
                str(self.output_dir / "yolov8n_detection.jpg"))
            
            results["face-detection"] = self.test_ssd_detection(
                "face-detection-retail-0005", input_path, 
                str(self.output_dir / "face_detection.jpg"),
                ["background", "face"])
            
            results["lp-detection"] = self.test_ssd_detection(
                "vehicle-license-plate-detection-barrier-0106", input_path, 
                str(self.output_dir / "lp_detection.jpg"),
                ["background", "vehicle", "license_plate"])
            
            results["lprnet"] = self.test_lprnet_recognition(input_path, 
                str(self.output_dir / "lprnet_recognition.jpg"))
        
        return results
    
    def download_all_models(self) -> bool:
        """Download all models"""
        models = AVAILABLE_MODELS
        success_count = 0
        
        for model_name in models:
            if self.download_model(model_name):
                success_count += 1
        
        console.print(f"[green]✅ Downloaded {success_count}/{len(models)} models successfully[/green]")
        return success_count == len(models)

# Model aliases for convenience
MODEL_ALIASES = {
    "face": "face-detection-retail-0005",
    "license_plate_detection": "vehicle-license-plate-detection-barrier-0106",
    "lp_detection": "vehicle-license-plate-detection-barrier-0106",
    "person_reid": "person-reidentification-retail-0286",
    "reid": "person-reidentification-retail-0286",
}

# All available models
AVAILABLE_MODELS = [
    "yolov8n",
    "face-detection-retail-0005",
    "vehicle-license-plate-detection-barrier-0106",
    "lprnet",
    "person-reidentification-retail-0286",
]

def main():
    parser = argparse.ArgumentParser(description="OpenVINO AI Model Testing Suite")
    model_choices = AVAILABLE_MODELS + list(MODEL_ALIASES.keys()) + ["all"]
    parser.add_argument("--model", choices=model_choices, 
                       default="all", help="Model to test (or alias: face, lp_detection, person_reid)")
    parser.add_argument("--input", type=str, help="Input image or video path")
    parser.add_argument("--output", type=str, help="Output path")
    parser.add_argument("--download-models", action="store_true", help="Download all models")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    
    args = parser.parse_args()
    
    # Resolve model alias
    model_name = MODEL_ALIASES.get(args.model, args.model)
    
    tester = OpenVINOModelTester()
    
    if args.list_models:
        # List available models
        table = Table(title="Available Models")
        table.add_column("Model Name", style="cyan")
        table.add_column("Alias", style="yellow")
        table.add_column("Type", style="magenta")
        table.add_column("Description", style="green")
        
        for m in AVAILABLE_MODELS:
            alias = next((k for k, v in MODEL_ALIASES.items() if v == m), "-")
            try:
                config = tester.load_model_config(m)
                table.add_row(m, alias, config.get("model_type", "unknown"), config.get("description", ""))
            except FileNotFoundError:
                table.add_row(m, alias, "Unknown", "Config not found")
        
        console.print(table)
        return
    
    if args.download_models:
        tester.download_all_models()
        return
    
    if not args.input:
        console.print("[red]❌ Please provide input path with --input[/red]")
        return
    
    if not os.path.exists(args.input):
        console.print(f"[red]❌ Input file not found: {args.input}[/red]")
        return
    
    # Run tests
    if model_name == "all":
        results = tester.run_all_tests(args.input)
        
        # Display results
        table = Table(title="Test Results")
        table.add_column("Model", style="cyan")
        table.add_column("Status", style="green")
        
        for m, success in results.items():
            status = "✅ PASSED" if success else "❌ FAILED"
            table.add_row(m, status)
        
        console.print(table)
    else:
        # Test specific model
        is_video = args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
        output_base = model_name.replace("-", "_")
        
        if is_video:
            # Video processing
            output_path = args.output or f"output/{output_base}_video_output.mp4"
            success = tester.test_video_processing(model_name, args.input, output_path)
        else:
            # Image processing based on model type
            output_path = args.output or f"output/{output_base}_output.jpg"
            
            if model_name == "yolov8n":
                success = tester.test_yolov8n_detection(args.input, output_path)
            elif model_name == "lprnet":
                success = tester.test_lprnet_recognition(args.input, output_path)
            elif model_name == "face-detection-retail-0005":
                success = tester.test_ssd_detection(model_name, args.input, output_path, ["background", "face"])
            elif model_name == "vehicle-license-plate-detection-barrier-0106":
                success = tester.test_ssd_detection(model_name, args.input, output_path, ["background", "vehicle", "license_plate"])
            elif model_name == "person-reidentification-retail-0286":
                console.print("[yellow]⚠️ Person re-ID model is for tracking, not standalone detection[/yellow]")
                console.print("[yellow]   Use with person detection + tracking pipeline[/yellow]")
                success = False
            else:
                console.print(f"[red]❌ Unknown model: {model_name}[/red]")
                success = False
        
        if success:
            console.print(f"[green]✅ {model_name} test completed successfully[/green]")
        else:
            console.print(f"[red]❌ {model_name} test failed[/red]")

if __name__ == "__main__":
    main()
