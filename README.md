# Atriva AI API with OpenVINO 🚀

This is a FastAPI-based AI API that leverages **OpenVINO** for optimized deep learning inference.  
It provides a RESTful interface for running AI models, such as object detection and image classification.

## **⚡ Features**
✅ FastAPI-based AI API  
✅ OpenVINO optimization for inference  
✅ Dockerized for easy deployment  
✅ Comprehensive testing suite  
✅ Multiple AI models supported:
   - YOLOv8n Object Detection
   - LPRNet License Plate Recognition
   - Vehicle Detection and Tracking  

## **📂 Project Structure**

```plaintext
atriva-ai-openvino/
│── app/
│   ├── routes.py         # API route definitions
│   ├── services.py       # AI model processing logic
│   ├── models.py         # Data models and schemas
│   ├── model_capabilities.py  # Model capabilities and metadata
│   ├── shared_data.py    # Shared data utilities
│── models/               # Pretrained OpenVINO models
│   ├── yolov8n/          # YOLOv8n object detection model
│   ├── lprnet/           # LPRNet license plate recognition model
│   └── vehicle_tracking/ # Vehicle detection and tracking model
│── tests/                # Comprehensive testing suite
│   ├── test_runner.py    # Main test runner with model download/conversion
│   ├── test_yolov8_openvino.py   # YOLOv8 detection (supports n/s/m sizes)
│   ├── test_vehicle_tracking.py  # Vehicle tracking (IoU + ByteTrack)
│   ├── test_images/      # Sample test images
│   ├── test_videos/      # Sample test videos
│   ├── output/           # Generated output files
│   └── requirements.txt  # Test dependencies
│── main.py               # Entry point for FastAPI
│── config.py             # Configuration settings
│── requirements.txt      # Python dependencies
│── Dockerfile            # Docker configuration
│── README.md             # Project documentation
```

---

## **🚀 Getting Started**

### **Overview**

This section covers everything you need to get the Atriva AI API up and running. Whether you're setting up locally, using Docker, or looking for the quickest way to start, you'll find the instructions here. The setup process includes cloning the repository, installing dependencies, downloading AI models, and running the service.

### **🔧 Setup & Installation**

#### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/atriva-ai/atriva-ai-openvino.git
cd atriva-ai-openvino
```

#### **2️⃣ Create a Virtual Environment**
```sh
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
```

#### **3️⃣ Download AI Models**
```sh
# Download all required model files
cd tests
python test_runner.py --download-models

# Or download individual models
python test_runner.py --model yolov8n --download
python test_runner.py --model lprnet --download
python test_runner.py --model vehicle_tracking --download
```

**📝 Important**: Model binary files (.pt, .bin, .xml) are not included in the repository due to size constraints. They will be downloaded automatically when needed.

#### **4️⃣ Run the API Locally**
```sh
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
Access the API documentation at:  
👉 **http://localhost:8000/docs**

### **🐳 Running with Docker**

#### **1️⃣ Build the Docker Image**
```sh
docker build -t atriva-ai-openvino .
```

#### **2️⃣ Run the Container**
```sh
docker run -d -p 8000:8000 --name ai-openvino-container atriva-ai-openvino
```
Now, visit:  
👉 **http://localhost:8000/docs**

### **🚀 Quick Start (Recommended)**

**Models are pre-built and ready to use:**

```sh
# 1. Build Docker image (models already included)
docker build -t atriva-ai-openvino .

# 2. Run AI service
docker run -d -p 8001:8001 --name ai-inference atriva-ai-openvino

# 3. Test API
curl http://localhost:8001/models
```

---

## **📥 Models**

### **Overview**

The Models section covers everything related to AI model management, including the build architecture, model conversion workflow, available models, and how to add new models. The system uses a pre-built model approach for optimal performance, with models converted to OpenVINO IR format for efficient inference.

### **🏗️ Model Build Architecture**

The AI service uses a **pre-built model approach** for optimal performance:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Development   │    │   Docker Build   │    │   Production    │
│   (Host)        │    │   (Container)    │    │   (Runtime)     │
├─────────────────┤    ├──────────────────┤    ├─────────────────┤
│ • Scripts       │    │ • Copy models    │    │ • Load models   │
│ • Model conv.   │───▶│ • Install deps   │───▶│ • Run inference │
│ • Testing       │    │ • Fast build     │    │ • API endpoints │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### **📋 Requirements Structure**

| File | Purpose | Python Version | Dependencies |
|------|---------|----------------|-------------|
| `requirements.txt` | **Docker AI Service** | 3.12 | FastAPI, OpenVINO, NumPy, OpenCV |
| `tests/requirements.txt` | **Testing Environment** | **3.11** | Testing framework, utilities |
| `scripts/requirements.txt` | **Model Conversion** | **3.11** | PyTorch, Ultralytics, OpenVINO |

### **📦 Model Management**

#### **Available Models:**
- ✅ **YOLOv8n** - Object detection (pre-built)
- ✅ **YOLOv8s** - Object detection (pre-built)  
- ✅ **YOLOv8m** - Object detection (pre-built)
- ✅ **LPRNet** - License plate recognition (pre-built)
- ✅ **Vehicle Tracking** - Vehicle detection (pre-built)

#### **Model Files Structure:**
```
models/
├── yolov8n/
│   ├── yolov8n.xml          # OpenVINO model structure
│   ├── yolov8n.bin          # OpenVINO model weights
│   └── model.json           # Model configuration
├── lprnet/
│   ├── lprnet.xml
│   ├── lprnet.bin
│   └── model.json
└── vehicle_tracking/
    ├── vehicle_tracking.xml
    ├── vehicle_tracking.bin
    └── model.json
```

#### **Verify Model Installation:**
```sh
# Check all models are present
ls models/*/model.json
ls models/*/*.xml
ls models/*/*.bin

# Expected output:
# models/lprnet/model.json models/vehicle_tracking/model.json models/yolov8n/model.json
# models/lprnet/lprnet.xml models/vehicle_tracking/vehicle_tracking.xml models/yolov8n/yolov8n.xml
# models/lprnet/lprnet.bin models/vehicle_tracking/vehicle_tracking.bin models/yolov8n/yolov8n.bin
```

#### **🔄 Adding New Models**

1. **Create model directory:**
   ```sh
   mkdir models/new_model
   ```

2. **Add model files:**
   - `new_model.xml` - OpenVINO structure
   - `new_model.bin` - OpenVINO weights  
   - `model.json` - Configuration

3. **Update model capabilities:**
   ```python
   # Add to app/model_capabilities.py
   ```

4. **Test the model:**
   ```sh
   python test_runner.py --model new_model --input test_images/sample.jpg
   ```

### **🤖 Available Models**

#### **YOLOv8n Object Detection**
- **Purpose**: Detect 80 COCO object classes
- **Input**: 640×640 RGB images
- **Output**: Bounding boxes with class labels and confidence scores (NMS applied)
- **Model Source**: Auto-downloaded from Ultralytics and converted to OpenVINO IR format
- **Use Cases**: General object detection, surveillance, autonomous vehicles

#### **LPRNet License Plate Recognition**
- **Purpose**: Recognize license plate text
- **Input**: 24×94 RGB images (cropped license plate regions)
- **Output**: Recognized text with character-level confidence
- **Note**: Requires pre-cropped license plate images (use detection model first)
- **Use Cases**: Parking management, traffic enforcement, access control

#### **Vehicle Detection and Tracking**
- **Purpose**: Detect and track vehicles/persons across video frames
- **Input**: 640×640 RGB images (uses YOLOv8n for detection)
- **Output**: Bounding boxes with unique persistent track IDs
- **Tracking Algorithms**:
  - **IoU Tracking**: Simple overlap-based matching
  - **ByteTrack**: Advanced algorithm with low-confidence detection recovery
- **Use Cases**: Traffic monitoring, parking analytics, fleet management

---

## **🔧 Development**

### **Overview**

The Development section provides information about the development workflow, including Python version requirements, environment setup for testing and model conversion, and best practices for working with the codebase.

### **🔧 Development Workflow**

#### **⚠️ Python Version Requirements**
- **Docker AI Service**: Python 3.12 (handled by Dockerfile)
- **Testing Environment**: **Python 3.11 required** (PyTorch compatibility)
- **Scripts Environment**: **Python 3.11 required** (PyTorch compatibility)

#### **For Testing & Development:**
```sh
# Setup test environment with Python 3.11
cd tests
pyenv local 3.11.13  # or use python3.11
python3.11 -m venv test-venv-py311
source test-venv-py311/bin/activate
pip install -r requirements.txt

# Run tests
python test_runner.py --model yolov8n --input test_images/sample.jpg
```

#### **For Model Conversion:**
```sh
# Setup scripts environment with Python 3.11
cd scripts
pyenv local 3.11.13  # or use python3.11
python3.11 -m venv scripts-venv-py311
source scripts-venv-py311/bin/activate
pip install -r requirements.txt

# Convert PyTorch to OpenVINO
python convert_to_openvino.py --size n
```

#### **Why Python 3.11?**
- **PyTorch compatibility**: PyTorch doesn't have pre-built wheels for Python 3.13
- **NumPy compatibility**: Avoids NumPy 2.x compatibility issues
- **Stable ML ecosystem**: Most ML libraries are tested with Python 3.11

---

## **🛠 API**

### **Overview**

The API section documents the RESTful endpoints available in the Atriva AI API. The service provides a FastAPI-based interface for running AI inference tasks, managing models, and accessing camera frames. All endpoints are documented with interactive Swagger UI available at `/docs` when the service is running.

The API is organized into several categories:
- **System & Health**: Basic health checks and system status
- **Model Information**: Query available models, objects, and capabilities
- **Model Management**: Load models with specific accelerators
- **Inference**: Run object detection and AI inference on images or camera frames
- **Camera & Shared Frames**: Access decoded camera frames and run inference on them

### **System & Health Endpoints**

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Root endpoint - returns API status message |
| `GET` | `/health` | Health check with shared volume status and available cameras |

**Example Response (`/health`):**
```json
{
  "status": "healthy",
  "shared_volumes": {
    "frames_path": "/path/to/frames",
    "frames_exists": true,
    "temp_path": "/path/to/temp",
    "temp_exists": true
  },
  "available_cameras": ["camera1", "camera2"]
}
```

### **Model Information Endpoints**

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/models` | List all available models |
| `GET` | `/objects` | List all available object types for detection |
| `GET` | `/model/info` | Get detailed model information, types, objects, accelerators, and architecture |
| `GET` | `/models/{model_name}/capabilities` | Get detailed capabilities for a specific model |

**Example Response (`/models`):**
```json
{
  "available_models": ["yolov8n", "yolov8s", "yolov8m", "vehicle", "car", "person", ...]
}
```

**Example Response (`/model/info`):**
```json
{
  "models": {...},
  "model_types": ["detection", "classification"],
  "objects": ["car", "person", "bicycle", ...],
  "accelerators": ["cpui8", "cpu16", "cpu32"],
  "architecture": "openvino"
}
```

**Available Accelerators:**
- `cpui8` - CPU with INT8 precision
- `cpu16` - CPU with FP16 precision
- `cpu32` - CPU with FP32 precision (default)

### **Model Management Endpoints**

| Method | Endpoint | Description | Parameters |
|--------|----------|-------------|------------|
| `POST` | `/model/load` | Load a model with a specific accelerator | `model_name` (required), `accelerator` (optional, default: "cpu32") |

**Example Request:**
```bash
curl -X POST "http://localhost:8000/model/load?model_name=yolov8n&accelerator=cpu32"
```

**Example Response:**
```json
{
  "model_name": "yolov8n",
  "accelerator": "cpu32",
  "architecture": "openvino",
  "status": "loaded"
}
```

### **Inference Endpoints**

| Method | Endpoint | Description | Parameters |
|--------|----------|-------------|------------|
| `POST` | `/inference/detection` | Detect objects in an uploaded image | `object_name` (query param), `image` (file upload) |
| `POST` | `/inference/direct` | Run direct inference using a specific model | `model_name` (query param), `image` (file upload) |
| `POST` | `/inference/latest-frame` | Run inference on the latest frame from a camera | `camera_id` (query param), `model_name` (query param), `accelerator` (optional) |
| `POST` | `/inference/background` | Start background inference on all frames from a camera | `camera_id` (query param), `model_name` (query param), `accelerator` (optional) |

**Example Request (`/inference/detection`):**
```bash
curl -X POST "http://localhost:8000/inference/detection?object_name=car" \
  -F "image=@test_image.jpg"
```

**Example Response:**
```json
{
  "objects": [
    {
      "class_id": 2,
      "confidence": 0.95,
      "bbox": [100, 150, 300, 400]
    }
  ]
}
```

**Example Request (`/inference/direct`):**
```bash
curl -X POST "http://localhost:8000/inference/direct?model_name=yolov8n" \
  -F "image=@test_image.jpg"
```

**Example Response:**
```json
{
  "model_name": "yolov8n",
  "input_shape": [1, 3, 640, 640],
  "output_shape": [1, 84, 8400],
  "output": [[...]]
}
```

### **Camera & Shared Frame Endpoints**

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/shared/cameras` | List all cameras that have decoded frames available |
| `GET` | `/shared/cameras/{camera_id}/frames` | Get information about decoded frames for a specific camera |
| `GET` | `/shared/cameras/{camera_id}/frames/latest` | Get the latest decoded frame image for a camera |
| `GET` | `/shared/cameras/{camera_id}/frames/{frame_index}` | Get a specific frame by index for a camera |
| `POST` | `/shared/cameras/{camera_id}/inference` | Run object detection on the latest frame from a camera | `object_name` (query param) |

**Example Request (`/shared/cameras`):**
```bash
curl "http://localhost:8000/shared/cameras"
```

**Example Response:**
```json
{
  "cameras": ["camera1", "camera2"]
}
```

**Example Request (`/shared/cameras/{camera_id}/inference`):**
```bash
curl -X POST "http://localhost:8000/shared/cameras/camera1/inference?object_name=car"
```

**Example Response:**
```json
{
  "camera_id": "camera1",
  "frame_path": "/path/to/frame.jpg",
  "object_name": "car",
  "detections": [
    {
      "class_id": 2,
      "confidence": 0.92,
      "bbox": [150, 200, 350, 450]
    }
  ]
}
```

### **API Documentation**

When the service is running, you can access:
- **Interactive API Documentation (Swagger UI)**: `http://localhost:8000/docs`
- **Alternative API Documentation (ReDoc)**: `http://localhost:8000/redoc`

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

---

## **🚨 Troubleshooting**

### **Overview**

The Troubleshooting section provides solutions to common issues you may encounter when setting up or using the Atriva AI API. This includes Python version compatibility problems, model download and conversion issues, and API usage errors. Each issue includes a description of the problem and step-by-step solutions.

### **Python Version Issues**

#### **Error: "No matching distribution found for torch"**
```bash
# Problem: Using Python 3.13
# Solution: Use Python 3.11
pyenv local 3.11.13
python3.11 -m venv venv-py311
source venv-py311/bin/activate
pip install -r requirements.txt
```

#### **Error: "NumPy compatibility issues"**
```bash
# Problem: NumPy 2.x with PyTorch
# Solution: Downgrade NumPy
pip install "numpy<2"
```

#### **Error: "ModuleNotFoundError: No module named 'ultralytics'"**
```bash
# Problem: Missing PyTorch dependencies
# Solution: Use Python 3.11 environment
cd scripts
pyenv local 3.11.13
python3.11 -m venv scripts-venv-py311
source scripts-venv-py311/bin/activate
pip install -r requirements.txt
```

### **Model Issues**

#### **Error: "Unable to read the model: model.xml"**
```bash
# Problem: Corrupted model files (HTML error page instead of XML)
# Solution: Re-download using test_runner.py which handles conversion properly
cd tests
python test_runner.py --download-models
```

#### **Error: "404 Not Found" when downloading models**
```bash
# Problem: Direct OpenVINO model URLs don't exist for YOLOv8
# Solution: Use ultralytics to download PyTorch weights and convert to OpenVINO
# test_runner.py handles this automatically:
python test_runner.py --download-models

# Manual conversion:
python -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # Downloads automatically
model.export(format='openvino')  # Converts to .xml/.bin
"
```

#### **Error: "Wrong class labels (class_34 instead of car)"**
```bash
# Problem: Loading classes from wrong metadata.yaml file
# Solution: Ensure ultralytics/metadata.yaml is present (has 80 COCO classes)
# The test scripts prioritize loading from models/yolov8n/ultralytics/metadata.yaml
```

#### **Error: "Incompatible inputs of type: ConstOutput"**
```bash
# Problem: Incorrect OpenVINO inference API usage
# Old incorrect code: result = model([input_tensor], {input_tensor: input_data})
# Correct code: result = model(input_data)
```

#### **Error: "too many values to unpack (expected 6)"**
```bash
# Problem: YOLOv8 output format is (1, 84, 8400) not (1, N, 6)
# Solution: Transpose output and parse correctly:
# predictions = outputs[0].T  # Shape: (8400, 84)
# First 4 values: cx, cy, w, h
# Remaining 80 values: class scores
```

### **Model File Structure**
Each model directory contains:
- `model.json` - Model configuration and metadata
- `metadata.yaml` - Additional model information
- `*.xml` - OpenVINO model structure (downloaded)
- `*.bin` - OpenVINO model weights (downloaded)
- `*.pt` - PyTorch model weights (optional, for conversion)

**Note**: Binary files (.pt, .bin, .xml) are not stored in Git due to size constraints. They are downloaded automatically when needed.

---

## **📜 License**

This project is licensed under the **MIT License**.
