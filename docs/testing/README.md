# 🏗️ Atriva AI OpenVINO Inference API — Development & Testing Guide

This document provides essential information for developers and testers working with the Atriva AI Runtime powered by OpenVINO.
You'll find details on API endpoints, development workflows, environment setup, model conversion, and automated testing for object detection and tracking tasks.

- **API Endpoints**: Reference for REST endpoints, system health, model info, and inference requests.
- **Development**: Environment setup, OpenVINO integration details, compatible Python versions, and workflow guidance.
- **Model Preparation**: Steps to export, convert, and quantize models for optimal device performance.
- **Testing**: How to run image and video inference, vehicle tracking, and generate performance reports.

> 💡 **Tip:** See each section for code snippets, CLI commands, and troubleshooting.

---

## **🧪 Testing**

### **Overview**

The Testing section covers how to run the comprehensive test suite for the AI models. This includes setup instructions, running tests on images and videos, and using different model sizes. The test suite supports YOLOv8 object detection, vehicle tracking with multiple algorithms, and provides detailed performance reports.

**⚠️ Prerequisites**: Make sure you have downloaded the required models first (see [Model Management](#-model-management) section above).

```sh
# Setup test environment
cd tests
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download/convert YOLOv8 models (auto-downloads from Ultralytics and converts to OpenVINO)
python test_runner.py --download-models
```

### **Image Detection**
```sh
# Test YOLOv8n on image
python test_runner.py --model yolov8n --input test_images/dog_bike_car.jpg

# Test with different model sizes
python test_yolov8_openvino.py --input test_images/sample.jpg --size n  # nano
python test_yolov8_openvino.py --input test_images/sample.jpg --size s  # small
python test_yolov8_openvino.py --input test_images/sample.jpg --size m  # medium
```

### **Video Detection**
```sh
# Process video with YOLOv8
python test_runner.py --model yolov8n --input test_videos/sample_traffic.mp4

# Advanced video options
python test_yolov8_openvino.py --input test_videos/sample_traffic.mp4 --video --inference-fps 1 --length 30
```

### **Vehicle Tracking**
```sh
# Simple IoU-based tracking (default)
python test_vehicle_tracking.py --input test_videos/sample_traffic.mp4 --video --tracker iou

# ByteTrack algorithm (better occlusion handling)
python test_vehicle_tracking.py --input test_videos/sample_traffic.mp4 --video --tracker bytetrack
```

**Sample Output Report:**
```
═══ Summary Report ═══
✅ Processed 300 frames
✅ Total processing time: 45.23s
✅ Overall FPS (including I/O): 6.63
✅ Inference FPS (model only): 28.45
✅ Avg inference time per frame: 35.15ms
✅ Total detections: 1250
✅ Average detections per frame: 4.17
✅ Maximum active tracks: 12
✅ Tracker: bytetrack
✅ Saved annotated video: output/vehicle_tracking_video_output.mp4
```